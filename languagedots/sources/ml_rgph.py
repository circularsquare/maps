"""Mali RGPH5 2022, mother tongue ("langue maternelle") by région -> data/normalized/ml.csv.

    python sources/ml_rgph.py [--fetch]

SOURCE. INSTAT, RGPH5 *Caractéristiques culturelles de la population* (70 pages), the volume
religiondots draws Mali's religion from. Live at
https://www.instat-mali.org/laravel-filemanager/files/shares/rgph/rapport-caracteristiques-culturelles-population-rgph5_rpgh.pdf
(no login). --fetch copies religiondots' verified download (read-only) when it is there, and
otherwise downloads it; the digest is pinned either way.

QUESTION (Tableau 1.02, PDF p26-27): "Quelle est la langue maternelle de [Nom] ?", asked of the
population aged 3 and over; one answer, 28 codes. The volume's definition (PDF p25): the first
language learnt in early childhood, the language spoken to the child at home. The census also
asked the three languages each person speaks most (Tableau 1.03, annex A07, "principale langue
parlée"); that is a use question and is not drawn.

TABLES READ (1-based PDF pages):
  annex A06, pp56-57   % by région: 23 rows (19 named languages, "Autre langue du Mali",
                       "Autre langue africaine", "Autre langue étrangère", "Autre langue non
                       africaine") x 20 régions + Total, two decimals; "Effectif" per région.
                       Shares are of those who answered: non-response is spread in proportion.
  annex A03, p53       national counts by sex for all 28 codes and ND (non-response).
  Tableau 3.01, p39    national counts with ND spread in proportion (the second table).

THE BUILD. Each région's count = A06 share x A06 Effectif, as printed, with no rake. The shares
are of those who answered and Effectif includes the 68,582 who did not (ND), so each région's
non-response is spread over its own answers, which is what the shares themselves do. A rake to
A03's national counts was tried and dropped: those counts spread ND evenly over the country,
while the regional table shows it sat in the north (check 9: Tamasheq takes about 40% of it),
so raking would have taken 3.4% off Tamasheq in every région to put it on Bambara.
A06's "Autre langue étrangère" is, nationally, A03's six named foreign languages (French,
English, German, Russian, Chinese, Spanish); each région's count of it is split among the six in
their national proportions, and those rows are `derived` (20,500 people, 89% French).

CHECKS (all must pass):
  1. the PDF is the pinned 70-page volume (digest, %%EOF)
  2. A06: 23 rows x 21 columns parsed; the row labels are the expected ones in order; every
     région's column sums to 100 within 0.12 (23 cells rounded to 0.005)
  3. A06 Effectif: the digits on the page are the pinned 21 figures; the régions sum to the
     printed total, 19,144,633 (which Tableau 3.02 also prints)
  4. A03: male + female = total on every row; the rows sum to the printed total 19,141,587
  5. A06's Total column = A03's counts over the non-ND total, within 0.006 (one row excepted,
     printed)
  6. A06's Total column against the régions' shares weighted by Effectif: within 0.15, the
     misses printed (they are the non-response, see 9)
  7. Tableau 3.01's counts = A03's with ND prorated, within 3 people (its "Langue étrangère"
     row printed, not asserted)
  8. the 20 régions join religiondots' ml_hexes units one to one (both ways)
  9. drawn minus A03's answered count, per language, is >= minus the rounding bound, and the
     excesses sum to ND plus the 3,046 by which Effectif's total exceeds A03's
"""
import argparse
import hashlib
import re
import shutil
import sys
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

RAW = HERE / "data" / "raw" / "ml"
OUT = HERE / "data" / "normalized" / "ml.csv"
NAME = "rapport-caracteristiques-culturelles-population-rgph5_rpgh.pdf"
PDF = RAW / NAME
RD_PDF = HERE.parent / "religiondots" / "data" / "raw" / "ml" / NAME
ORIGINAL = "https://www.instat-mali.org/laravel-filemanager/files/shares/rgph/" + NAME
PDF_BYTES = 17_708_018
SHA256 = "57a1967d783b78e0b78208584312372ab06d3ba85d75d54119e1ecd7e13f3a9d"
PAGES = 70

P_A06 = (55, 56)       # 0-based
P_A03 = 52
P_T301 = 38

REGIONS = ["Kayes", "Kita", "Nioro", "Koulikoro", "Dioïla", "Nara", "Sikasso", "Bougouni",
           "Koutiala", "Ségou", "San", "Mopti", "Bandiagara", "Douentza", "Tombouctou", "Gao",
           "Kidal", "Taoudenni", "Ménaka", "Bamako"]
# A06 prints "Taoudénit" and "Dioila"; the hex layer has "Taoudenni" and "Dioïla"
A06_HEAD = ["Kayes", "Kita", "Nioro", "Koulikoro", "Dioila", "Nara", "Sikasso", "Bougouni",
            "Koutiala", "Ségou", "San", "Mopti", "Bandiagara", "Douentza", "Tombouctou", "Gao",
            "Kidal", "Taoudénit", "Ménaka", "Bamako", "Total"]

ROWS = ["Bambara/Bamanankan", "Malinké/Maninkakan", "Peulh/Fulfulde", "Sonrhai/Songhoy/Zarma",
        "Sarakole/Sooninke", "Khassonké/Xhassonkakan", "Sénoufo/Syenara", "Dogon/Dôgôsô",
        "Maure/Hasaniya", "Tamasheq", "Bobo/Bomu", "Kunabere", "Dafing", "Minianka/Mamara",
        "Haoussa", "Mossi/Moré", "Samogo/Dungooma", "Bozo/Tyako", "Arabe",
        "Autre langue du Mali", "Autre langue africaine", "Autre langue étrangère",
        "Autre langue non africaine"]
FOREIGN = ["Français", "Anglais", "Allemand", "Russe", "Chinois", "Espagnol"]
ETRANGERE = "Autre langue étrangère"
A03_ROWS = ROWS[:21] + FOREIGN + ["Autre langue non africaine", "ND"]

# A06's Effectif row, transcribed; check 3 asserts the page's digits are exactly these
EFFECTIF = [1638558, 599284, 596538, 2004358, 606102, 247549, 1360644, 1390113, 1027299,
            1985705, 734708, 763379, 646369, 134127, 634866, 609607, 72163, 92353, 203411,
            3797499]
EFFECTIF_TOTAL = 19_144_633
A03_TOTAL = 19_141_587

PCT2 = re.compile(r"^\d{1,3},\d\d$")
PCT1 = re.compile(r"^\d{1,3},\d$")
PCT0 = re.compile(r"^\d{1,3},$")
DIGIT = re.compile(r"^\d$")
TWO = re.compile(r"^\d\d$")
INT = re.compile(r"^\d+$")


def norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def sha(body):
    return hashlib.sha256(body).hexdigest()


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    if PDF.exists() and PDF.stat().st_size == PDF_BYTES:
        print("already have", PDF)
        return
    if RD_PDF.exists() and RD_PDF.stat().st_size == PDF_BYTES:
        shutil.copyfile(RD_PDF, PDF)
        print(f"copied religiondots' verified download -> {PDF}")
        return
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
    with urllib.request.urlopen(urllib.request.Request(ORIGINAL, headers=ua), timeout=600) as r:
        body = r.read()
    if body[:4] != b"%PDF" or not body.rstrip().endswith(b"%%EOF"):
        raise SystemExit(f"bad or truncated PDF ({len(body):,} bytes)")
    PDF.write_bytes(body)
    print(f"wrote {PDF} ({len(body):,} bytes)")


def tokens(doc, pno):
    return doc.load_page(pno).get_text().split()


def read_a06(doc):
    """Return ({row: [21 shares]}, effectif digit string, the label text in order)."""
    toks = tokens(doc, P_A06[0]) + tokens(doc, P_A06[1])
    nums, words, eff, mode = [], [], [], "pct"
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "A07":
            break
        if t == "Effectif":
            mode = "eff"
        elif mode == "pct" and PCT2.match(t):
            nums.append(float(t.replace(",", ".")))
        elif mode == "pct" and PCT1.match(t) and i + 1 < len(toks) and DIGIT.match(toks[i + 1]):
            nums.append(float((t + toks[i + 1]).replace(",", ".")))   # a cell broken over lines
            i += 1
        elif mode == "pct" and PCT0.match(t) and i + 1 < len(toks) and TWO.match(toks[i + 1]):
            nums.append(float((t + toks[i + 1]).replace(",", ".")))   # "100," + "00"
            i += 1
        elif mode == "eff" and INT.match(t):
            eff.append(t)
        elif mode == "pct":
            words.append(t)
        i += 1
    n = len(A06_HEAD)
    assert len(nums) == (len(ROWS) + 1) * n, (len(nums), (len(ROWS) + 1) * n)
    table = {r: nums[k * n:(k + 1) * n] for k, r in enumerate(ROWS)}
    ensemble = nums[len(ROWS) * n:]
    assert all(abs(x - 100) < 1e-9 for x in ensemble), ensemble
    return table, "".join(eff), norm(" ".join(words))


def read_counts(doc, pno, width):
    """Rows of a label (one or more lines) followed by `width` numbers ("9 555 469" is one)."""
    lines = [ln.strip() for ln in doc.load_page(pno).get_text().splitlines() if ln.strip()]
    rows, label, vals = [], [], []
    for ln in lines:
        if re.fullmatch(r"\d{1,3}(?: \d{3})*|\d+", ln):
            vals.append(int(ln.replace(" ", "")))
        elif re.fullmatch(r"\d{1,3},\d\d", ln):
            vals.append(ln)
        elif re.fullmatch(r"[\d ]+ 100,00", ln):        # "9 555 471 100,00"
            vals += [int(ln[:-7].replace(" ", "")), "100,00"]
        else:
            if vals:
                rows.append((label[-1], vals))
                label, vals = [], []
            label.append(ln)
    if vals:
        rows.append((label[-1], vals))
    # a row's label is the line just above its numbers (every label here fits on one line)
    return {lab: v for lab, v in rows if len(v) == width}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    import fitz
    import pyogrio

    body = PDF.read_bytes()
    assert len(body) == PDF_BYTES and body.rstrip().endswith(b"%%EOF"), len(body)
    doc = fitz.open(PDF)
    assert doc.page_count == PAGES, doc.page_count
    print(f"1. {NAME}: {PAGES} pages, {len(body):,} bytes, sha256 {sha(body)[:16]}..., %%EOF ok")
    assert sha(body) == SHA256, sha(body)

    # ---- 2. A06 shares
    a06, effdigits, text = read_a06(doc)
    pos = 0
    for lab in ROWS:
        k = text.find(norm(lab), pos)
        assert k >= 0, f"A06 row label {lab!r} not found in order"
        pos = k + len(norm(lab))
    head = "".join(norm(h) for h in A06_HEAD)
    # the header is broken mid-word ("Kaye s"), so compare it with the spaces gone
    assert head in text, "A06 column header"
    regions = A06_HEAD[:20]
    sums = [sum(a06[r][j] for r in ROWS) for j in range(21)]
    worst = max(abs(s - 100) for s in sums)
    assert worst <= 0.12, sums
    print(f"2. A06: {len(ROWS)} rows x 21 columns, labels in order; columns sum to 100 within "
          f"{worst:.2f}")

    # ---- 3. Effectif
    assert effdigits == "".join(map(str, EFFECTIF)) + str(EFFECTIF_TOTAL), effdigits
    assert abs(sum(EFFECTIF) - EFFECTIF_TOTAL) <= 1, sum(EFFECTIF)
    print(f"3. A06 Effectif: the page's digits are the 20 pinned figures + {EFFECTIF_TOTAL:,}; "
          f"they sum to {sum(EFFECTIF):,} (weighted counts, rounded)")

    # ---- 4. A03
    a03raw = read_counts(doc, P_A03, 3)
    a03, off1 = {}, []
    for lab in A03_ROWS + ["Total"]:
        hit = [k for k in a03raw if norm(k) == norm(lab)]
        assert len(hit) == 1, (lab, list(a03raw))
        m, f, t = a03raw[hit[0]]
        assert abs(m + f - t) <= 1, (lab, m, f, t)     # weighted counts, rounded
        if m + f != t:
            off1.append(lab)
        a03[lab] = t
    total = a03.pop("Total")
    assert total == A03_TOTAL and abs(sum(a03.values()) - total) <= 3, (total, sum(a03.values()))
    nd = a03["ND"]
    answered = total - nd
    print(f"4. A03: {len(a03)} rows, M + F = total on each (off by one: {off1}); rows sum to "
          f"{sum(a03.values()):,} against the printed {total:,}; ND {nd:,} "
          f"({100 * nd / total:.2f}%)")
    total = sum(a03.values())   # the margins must agree exactly, or the rake never converges

    # national counts in A06's rows, ND prorated
    nat06 = {r: a03[r] for r in ROWS if r != ETRANGERE}
    nat06[ETRANGERE] = sum(a03[f] for f in FOREIGN)

    # ---- 5. A06 Total column vs A03
    off = {r: round(100 * nat06[r] / answered, 2) - a06[r][20] for r in ROWS}
    bad = {r: d for r, d in off.items() if abs(d) > 0.006}
    print(f"5. A06 Total = A03 / {answered:,} for {len(ROWS) - len(bad)} of {len(ROWS)} rows; "
          f"off: " + ", ".join(f"{r} {100 * nat06[r] / answered:.3f} printed {a06[r][20]:.2f}"
                               for r in bad))
    # Tableau 3.01 prints Malinké 7.11 and Sonrhai 4.57 from the same counts; A06's Total
    # column is a hundredth off on three rows, so it looks closed to 100.00 by hand
    assert len(bad) <= 3 and all(abs(d) <= 0.011 for d in bad.values()), bad

    # ---- 6. weighted régions vs Total
    # Not exact, and the misses have a pattern: the north's languages come out high (Tamasheq
    # 3.99 against 3.85, Songhay, Hassaniya, Arabic), the rest low. The shares are of those who
    # answered while Effectif includes the non-response, so this is what non-response
    # concentrated in the north does (check 9 says how much). Asserted loosely.
    wd = {r: sum(a06[r][j] * EFFECTIF[j] for j in range(20)) / sum(EFFECTIF) - a06[r][20]
          for r in ROWS}
    w = max(map(abs, wd.values()))
    print(f"6. A06 régions weighted by Effectif against its Total column: within {w:.3f}; "
          + ", ".join(f"{r.split('/')[0]} {a06[r][20] + d:.3f} ({a06[r][20]:.2f})"
                      for r, d in wd.items() if abs(d) > 0.01))
    assert w <= 0.15, w

    # ---- 7. Tableau 3.01
    t301 = read_counts(doc, P_T301, 6)
    prorated = {r: a03[r] * total / answered for r in ROWS[:20]}
    # 3.01 spells several labels differently (Sonhrai, Sarakolé, Senoufo, Gôgôsô): same order
    labs = list(t301)
    assert len(labs) == 21 and norm(labs[-1]) == "langueetrangere", labs
    worst7 = 0
    for r, k in zip(ROWS[:20], labs):
        assert norm(r)[:3] == norm(k)[:3], (r, k)
        worst7 = max(worst7, abs(t301[k][4] - prorated[r]))
    assert worst7 <= 3, worst7
    le = t301[[k for k in t301 if norm(k) == "langueetrangere"][0]][4]
    rest = sum(a03[r] for r in ROWS[20:] if r != ETRANGERE) + nat06[ETRANGERE]
    print(f"7. Tableau 3.01 = A03 with ND prorated for the 20 named rows, within {worst7:.1f}; "
          f"its 'Langue étrangère' {le:,} against A03's four remainders prorated "
          f"{rest * total / answered:,.0f} (it holds 'Autre langue non africaine' twice: "
          f"{(rest + a03['Autre langue non africaine']) * total / answered:,.0f})")

    # ---- 8. join
    units = sorted(pyogrio.read_dataframe(RD_GEO / "ml" / "ml_hexes.gpkg", columns=["unit"],
                                          read_geometry=False)["unit"].unique())
    assert sorted(REGIONS) == units, set(REGIONS) ^ set(units)
    for h, u in zip(A06_HEAD, REGIONS):
        assert norm(h) == norm(u) or (h, u) == ("Taoudénit", "Taoudenni"), (h, u)
    print("8. A06's 20 régions = religiondots' ml_hexes units, one to one")

    # ---- the counts
    # ---- 9. the counts: A06's share x A06's Effectif, as printed, no rake. The non-response
    # stays in the région it was in, spread over that région's answers as the shares do.
    m = pd.DataFrame({r: [a06[r][j] / 100 * EFFECTIF[j] for j in range(20)] for r in ROWS},
                     index=REGIONS)
    excess = m.sum() - pd.Series(nat06)[ROWS]
    # each région's cell is rounded to 0.005 pp, so a language's sum can be off by this much
    tol = 0.005 / 100 * sum(EFFECTIF)
    print(f"9. drawn (share x Effectif) minus A03's answered count, by language; the excess is "
          f"the non-response the shares spread, so it should be >= -{tol:,.0f} (rounding):")
    for r in ROWS:
        print(f"     {r:<30} {m[r].sum():>12,.0f} {excess[r]:>+10,.0f}")
    print(f"   total excess {excess.sum():+,.0f} against A03's ND {nd:,} + the "
          f"{sum(EFFECTIF) - total:,} Effectif holds over A03's total")
    assert (excess >= -tol).all(), excess[excess < -tol]
    assert abs(excess.sum() - (nd + sum(EFFECTIF) - total)) <= 50, excess.sum()
    north = ["Tombouctou", "Gao", "Kidal", "Taoudenni", "Ménaka"]
    print(f"   Tamasheq's excess {excess['Tamasheq']:+,.0f} is "
          f"{100 * excess['Tamasheq'] / excess.sum():.0f}% of the non-response, where Tamasheq "
          f"is {100 * m['Tamasheq'].sum() / m.values.sum():.1f}% of the people: the "
          f"non-response sat in the north ({', '.join(north)} hold "
          f"{100 * m.loc[north].values.sum() / m.values.sum():.1f}% of Effectif)")

    rows = []
    fsum = sum(a03[f] for f in FOREIGN)
    for u in REGIONS:
        for r in ROWS:
            if r == ETRANGERE:
                for f in FOREIGN:
                    rows.append(dict(geo_id=u, geo_level="region", geo_name=u, source_category=f,
                                     pct=None, count=m.at[u, r] * a03[f] / fsum, tier="derived"))
            else:
                rows.append(dict(geo_id=u, geo_level="region", geo_name=u, source_category=r,
                                 pct=a06[r][REGIONS.index(u)], count=m.at[u, r],
                                 tier="measured"))
    df = pd.DataFrame(rows)
    df["count"] = df["count"].round().astype(int)
    natrows = pd.DataFrame([dict(geo_id="ML", geo_level="national", geo_name="Mali",
                                 source_category=k, pct=round(100 * v / total, 3), count=v,
                                 tier="universe") for k, v in a03.items()])
    out = pd.concat([df, natrows], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(df):,} région rows ({(df['count'] > 0).sum():,} non-zero), "
          f"{len(natrows)} national rows (A03 as printed); régions sum to {df['count'].sum():,} "
          f"against {total:,}")


if __name__ == "__main__":
    main()
