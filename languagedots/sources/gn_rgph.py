"""Guinea RGPH 2014, main national language spoken, by région -> data/normalized/gn.csv,
plus CLEAR Global's prefecture shares of the same census -> data/normalized/gn_clear.csv.

    python sources/gn_rgph.py [--fetch]

SOURCE. INS Guinée, RGPH 2014 (RGPH3), *État et structure de la population* (122 pp), the
volume religiondots draws Guinea's religion from (religiondots/sources/gn.md). Live at
https://www.stat-guinee.org/images/Documents/Publications/INS/rapports_enquetes/RGPH3/RGPH3_etat_structure.pdf
(also in the Wayback Machine). --fetch copies religiondots' verified download (read-only) when
it is there, and otherwise downloads it; the digest is pinned either way.

QUESTION (definitions, PDF p27): "Langue nationale parlée : il s'agit de la langue nationale
habituellement parlée par l'individu même s'il parle d'autres langues", one answer, people aged
3 and over (Tableau 5.07's title and total). The answers are Guinea's national languages only,
plus "Aucune" (no national language) and "Autre langue nationale". Main language, so `how`
says that.

TABLES READ (1-based PDF pages):
  Tableau 5.07, p88   % by urban/rural x sex, 24 rows; the Total column and the 3+ Effectif
                      9,439,468 (the national check).
  Tableau 5.08, p89   % by région, the same 24 rows x 8 régions + Total, one decimal, and an
                      Effectif per région. DRAWN. Its title says "plus de 15 ans", but its
                      Effectif total is 5.07's 3+ total to the person, and religiondots found
                      the same page's label wrong (religiondots/sources/gn.md §6).

KINDIA'S EFFECTIF IS MISPRINTED. 5.08 prints 140 044 for Kindia, a tenth of what it must be:
the régions then sum to 8,179,028 against the printed 9,439,468. The residual, 1,400,484, is
used: it is 0.899 of Kindia's resident population (Tableau 2.07, 1,558,109), the same ratio as
Boké and N'Zérékoré (0.899), inside the others' 0.877-0.911 (check 4). The misprint drops one
digit and shuffles the others ("140 044" against "1 400 484"), so the residual is taken, not a
guessed digit.

THE BUILD. Each région's count = 5.08 share x Effectif. One-decimal shares on régions of
0.66 to 1.73 million people, so a cell is good to about +-800 people, and a language printed
0,0 in a région is not drawn there.

CLEAR GLOBAL (https://data.humdata.org/dataset/guinea-languages, CC BY-SA 4.0): "main language
spoken in the household" proportions for the 8 régions and the 34 prefectures (33 plus
Conakry), made by CLEAR from the IPUMS 10% sample of this same census (variable LANGGN2). Used
ONLY to place each région's speakers among its prefectures (countries/gn.py); the counts are
INS's. Its codes are COD-AB's pcodes. Its 21 named languages are Glottolog names for 21 of INS's
24 rows; "Tomamania" (Manya), "Autre langue nationale" and "Aucune" are not among them and sit,
with children under 3, in CLEAR's `Unknown` (Macenta, where Manya is spoken, has 22.8% Unknown
against 8.5-12% everywhere else but Kérouané's 14.4%; check 8).

CHECKS (all must pass):
  1. the PDF is the pinned 122-page volume (digest, %%EOF)
  2. Tableau 5.08 parsed off the page = the transcription below, 24 rows x 9 columns
  3. every région column of 5.08 sums to 100 within 0.6 (24 one-decimal cells)
  4. Effectif: the seven printed régions + the Kindia residual = 9,439,468; each région's
     Effectif over its Tableau 2.07 resident population is 0.87-0.92 (Kankan, the youngest, 0.877)
  5. 5.08's Total column = 5.07's Total column (parsed), and the régions' shares weighted by
     Effectif reproduce it within 0.15
  6. the 8 région names join religiondots' gn_hexes units one to one (both ways)
  7. CLEAR: 34 prefectures whose pcodes are COD-AB's gin_admin2 pcodes, both ways; each
     prefecture's proportions sum to 1
  8. CLEAR's région shares (Unknown left out) against 5.08, for every language both name:
     within 3.5 points (a 10% sample of the same census; CLEAR's shares leave out the three
     rows it has no code for, so run a little high; worst is Susu in Boké, 35.4 against 32.2)
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

RAW = HERE / "data" / "raw" / "gn"
OUT = HERE / "data" / "normalized" / "gn.csv"
OUT_CLEAR = HERE / "data" / "normalized" / "gn_clear.csv"
NAME = "RGPH3_etat_structure.pdf"
PDF = RAW / NAME
RD_PDF = HERE.parent / "religiondots" / "data" / "raw" / "gn" / NAME
RD_ADM2 = HERE.parent / "religiondots" / "data" / "raw" / "gn" / "shp" / "gin_admin2.shp"
ORIGINAL = ("https://www.stat-guinee.org/images/Documents/Publications/INS/rapports_enquetes/"
            "RGPH3/" + NAME)
WAYBACK = ("http://web.archive.org/web/20200921185302id_/http://www.stat-guinee.org/images/"
           "Documents/Publications/INS/rapports_enquetes/RGPH3/" + NAME)
PDF_BYTES = 3_762_432
SHA256 = "76e06c3ba6cac3130319bffe1aa8889c1d5b43b5587ced5244c423e78df1264a"
PAGES = 122
P_T507, P_T508 = 87, 88    # 0-based

CLEAR_BASE = "https://data.humdata.org/dataset/0e0c217a-7764-41fc-bba9-ddbc1a40bb8f/resource/"
CLEAR = {
    0: CLEAR_BASE + "635e9c71-1f2c-432b-8652-c0bdba647b5e/download/clearglobal_language_use_gin_admin0.csv",
    1: CLEAR_BASE + "0264c6da-f76b-4339-b707-5a8b07f0de84/download/clearglobal_language_use_gin_admin1.csv",
    2: CLEAR_BASE + "0ab33c4b-052f-419d-b07f-1c8064d32347/download/clearglobal_language_use_gin_admin2.csv",
}

REGIONS = ["Boké", "Conakry", "Faranah", "Kankan", "Kindia", "Labé", "Mamou", "N'Zérékoré"]
CLEAR_ADM1 = {"GN001": "Boké", "GN002": "Conakry", "GN003": "Faranah", "GN004": "Kankan",
              "GN005": "Kindia", "GN006": "Labé", "GN007": "Mamou", "GN008": "N'Zérékoré"}

# Tableau 5.08 as printed: row -> 8 régions + Total. Check 2 asserts the page says exactly this.
ROWS = ["Aucune", "Soussou", "Poular", "Maninka", "Diakanka", "Baga", "Nalou", "Mikiforè",
        "Landouma", "Badiaranké", "Bassari", "Koniagui", "Djalonké", "Sarakolé/Maraka",
        "Kouranko", "Kissi", "Lélé", "Toma", "Koniaka", "Tomamania", "Kpèlè", "Mano", "Kono",
        "Autre langue nationale"]
T508 = {
    "Aucune":           (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.2),
    "Soussou":          (32.2, 37.0, 0.4, 0.3, 54.9, 0.3, 1.9, 0.3, 17.7),
    "Poular":           (45.8, 34.0, 27.9, 3.9, 35.2, 94.5, 92.4, 3.5, 34.6),
    "Maninka":          (2.0, 18.7, 36.3, 87.1, 5.3, 0.9, 4.3, 9.2, 24.9),
    "Diakanka":         (5.8, 1.3, 0.2, 0.1, 0.5, 1.7, 0.2, 0.1, 1.1),
    "Baga":             (2.3, 0.5, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.3),
    "Nalou":            (0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1),
    "Mikiforè":         (2.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.2),
    "Landouma":         (3.9, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
    "Badiaranké":       (0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1),
    "Bassari":          (0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1),
    "Koniagui":         (0.8, 0.4, 0.0, 0.1, 0.1, 0.0, 0.0, 0.4, 0.2),
    "Djalonké":         (0.3, 0.5, 6.7, 0.2, 0.2, 1.2, 0.0, 0.1, 0.9),
    "Sarakolé/Maraka":  (0.5, 0.1, 0.0, 0.0, 0.2, 0.7, 0.0, 0.0, 0.2),
    "Kouranko":         (0.0, 0.2, 12.1, 3.9, 0.1, 0.0, 0.0, 0.3, 1.9),
    "Kissi":            (0.3, 1.7, 13.2, 0.6, 1.0, 0.1, 0.2, 15.4, 4.1),
    "Lélé":             (0.0, 0.1, 2.3, 0.1, 0.1, 0.0, 0.0, 0.9, 0.4),
    "Toma":             (0.2, 0.8, 0.2, 0.2, 0.4, 0.0, 0.0, 7.8, 1.4),
    "Koniaka":          (0.1, 1.6, 0.1, 2.6, 0.3, 0.0, 0.0, 24.4, 4.5),
    "Tomamania":        (0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 4.5, 0.7),
    "Kpèlè":            (0.3, 1.4, 0.3, 0.3, 0.7, 0.2, 0.2, 23.4, 4.0),
    "Mano":             (0.1, 0.2, 0.0, 0.1, 0.1, 0.0, 0.0, 4.0, 0.7),
    "Kono":             (0.1, 0.3, 0.0, 0.1, 0.2, 0.0, 0.1, 4.9, 0.8),
    "Autre langue nationale": (1.1, 0.6, 0.1, 0.1, 0.2, 0.0, 0.4, 0.6, 0.4),
}
# 5.08's Effectif as printed; Kindia's 140 044 is the misprint (see the docstring)
EFFECTIF_PRINTED = {"Boké": 971_533, "Conakry": 1_507_729, "Faranah": 851_288,
                    "Kankan": 1_727_195, "Kindia": 140_044, "Labé": 898_034,
                    "Mamou": 664_849, "N'Zérékoré": 1_418_356}
EFFECTIF_TOTAL = 9_439_468
# Tableau 2.07, resident population by région (religiondots/sources/gn.py, its check 2)
RESIDENT = {"Boké": 1_080_948, "Conakry": 1_657_702, "Faranah": 938_925, "Kankan": 1_968_388,
            "Kindia": 1_558_109, "Labé": 992_339, "Mamou": 729_637, "N'Zérékoré": 1_577_086}

# INS row -> CLEAR's Glottolog code (the placement weight inside a région; None = population)
CLEAR_CODE = {
    "Soussou": "susu1250", "Poular": "pula1262", "Maninka": "mane1267",
    "Diakanka": "jaha1245", "Baga": "temn1245", "Nalou": "nalu1240", "Mikiforè": "mixi1241",
    "Landouma": "land1256", "Badiaranké": "bady1239", "Bassari": "bass1258",
    "Koniagui": "wame1240", "Djalonké": "yalu1240", "Sarakolé/Maraka": "soni1259",
    "Kouranko": "kura1250", "Kissi": "kiss1245", "Lélé": "lele1266", "Toma": "toma1245",
    "Koniaka": "kony1250", "Tomamania": "Unknown", "Kpèlè": "guin1254", "Mano": "mann1248",
    "Kono": "kono1267", "Aucune": None, "Autre langue nationale": None,
}

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r:
        return r.read()


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    if PDF.exists() and PDF.stat().st_size == PDF_BYTES:
        print("already have", PDF)
    elif RD_PDF.exists() and RD_PDF.stat().st_size == PDF_BYTES:
        shutil.copyfile(RD_PDF, PDF)
        print(f"copied religiondots' verified download -> {PDF}")
    else:
        for url in (ORIGINAL, WAYBACK):
            try:
                body = get(url)
            except Exception as e:  # noqa: BLE001
                print(f"  {url}: {e}")
                continue
            if body[:4] == b"%PDF" and body.rstrip().endswith(b"%%EOF"):
                PDF.write_bytes(body)
                print(f"wrote {PDF} ({len(body):,} bytes)")
                break
        else:
            raise SystemExit("could not fetch the RGPH 2014 structure volume")
    for level, url in CLEAR.items():
        out = RAW / f"clearglobal_gin_admin{level}.csv"
        body = get(url)
        if not body.startswith(b"location_code"):
            raise SystemExit(f"{url}: not CLEAR's CSV")
        out.write_bytes(body)
        print(f"wrote {out} ({len(body):,} bytes)")


def pct(t):
    return float(t.replace(",", "."))


def read_t508(doc):
    lines = [ln.strip() for ln in doc.load_page(P_T508).get_text().splitlines() if ln.strip()]
    start = lines.index("Aucune")
    rows, label, vals = {}, [], []
    for ln in lines[start:]:
        if ln.startswith("Total"):
            break
        if re.fullmatch(r"\d{1,3},\d", ln):
            vals.append(pct(ln))
            if len(vals) == 9:
                rows[" ".join(label)] = tuple(vals)
                label, vals = [], []
        else:
            label.append(ln)
    eff = lines[lines.index("Effectif") + 1:]
    eff_txt = " ".join(eff[:1])
    return rows, eff_txt


def read_t507_total(doc):
    """Tableau 5.07's last column (Total, both sexes), row by row."""
    lines = [ln.strip() for ln in doc.load_page(P_T507).get_text().splitlines() if ln.strip()]
    start = lines.index("Aucune")
    out, vals, label = [], [], []
    for ln in lines[start:]:
        if ln.startswith("Total"):
            break
        if re.fullmatch(r"\d{1,3},\d", ln):
            vals.append(pct(ln))
            if len(vals) == 9:
                out.append(vals[-1])
                vals, label = [], []
        else:
            label.append(ln)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()

    import fitz
    body = PDF.read_bytes()
    doc = fitz.open(PDF)
    say(len(body) == PDF_BYTES and hashlib.sha256(body).hexdigest() == SHA256
        and body.rstrip().endswith(b"%%EOF") and doc.page_count == PAGES,
        f"1. {NAME}: {len(body):,} bytes, sha256 pinned, %%EOF, {doc.page_count} pages")

    # 2. the page = the transcription
    parsed, eff_txt = read_t508(doc)
    fold = {norm(k): v for k, v in parsed.items()}
    want = {norm(k): v for k, v in T508.items()}
    # the page truncates "Sarakolé/Maraka" to "Sarakolé/Marak"
    fold[norm("Sarakolé/Maraka")] = fold.pop(norm("Sarakolé/Marak"))
    say(fold == want and len(parsed) == 24,
        f"2. Tableau 5.08: {len(parsed)} rows x 9 parsed off p89 = the transcription")

    # 3. columns sum to 100
    sums = [sum(T508[r][j] for r in ROWS) for j in range(9)]
    say(all(abs(s - 100) <= 0.6 for s in sums),
        "3. 5.08 columns sum to 100 within 0.6: " + " ".join(f"{s:.1f}" for s in sums))

    # 4. Effectif and the Kindia misprint
    digits = re.sub(r"\D", "", eff_txt)
    want_digits = "".join(str(EFFECTIF_PRINTED[r]) for r in REGIONS) + str(EFFECTIF_TOTAL)
    say(digits == want_digits, f"4a. the Effectif row's digits are the 8 printed figures + "
        f"{EFFECTIF_TOTAL:,} (Kindia printed {EFFECTIF_PRINTED['Kindia']:,})")
    eff = dict(EFFECTIF_PRINTED)
    eff["Kindia"] = EFFECTIF_TOTAL - sum(v for r, v in eff.items() if r != "Kindia")
    ratios = {r: eff[r] / RESIDENT[r] for r in REGIONS}
    say(eff["Kindia"] == 1_400_484 and all(0.87 <= x <= 0.92 for x in ratios.values()),
        f"4b. Kindia's Effectif = the residual {eff['Kindia']:,}; Effectif / resident: "
        + ", ".join(f"{r} {x:.3f}" for r, x in ratios.items()))

    # 5. Total column = 5.07's; régions weighted reproduce it
    t507 = read_t507_total(doc)
    say(t507 == [T508[r][8] for r in ROWS], "5a. 5.08's Total column = 5.07's Total column")
    worst = 0.0
    for r in ROWS:
        w = sum(T508[r][j] * eff[REGIONS[j]] for j in range(8)) / EFFECTIF_TOTAL
        worst = max(worst, abs(w - T508[r][8]))
    say(worst <= 0.15, f"5b. régions weighted by Effectif vs the Total column: worst {worst:.3f}")

    # 6. join
    import geopandas as gpd
    hexes = gpd.read_file(RD_GEO / "gn" / "gn_hexes.gpkg", ignore_geometry=True)
    units = set(hexes["unit"].astype(str))
    say(units == set(REGIONS), f"6. the 8 régions = religiondots' gn_hexes units ({len(units)})")

    # 7. CLEAR prefectures
    c2 = pd.read_csv(RAW / "clearglobal_gin_admin2.csv")
    adm2 = gpd.read_file(RD_ADM2, engine="fiona", ignore_geometry=True)
    pc = set(c2["location_code"])
    sums2 = c2.groupby("location_code")["proportion_value"].sum()
    say(len(pc) == 34 and pc == set(adm2["adm2_pcode"]) and (sums2 - 1).abs().max() < 1e-6,
        f"7. CLEAR: {len(pc)} prefectures = COD-AB gin_admin2 pcodes both ways; shares sum to 1")
    say(set(CLEAR_CODE.values()) - {None} <= set(c2["language_code"]),
        "7b. every CLEAR code the placement uses is in the admin2 file")

    # 8. CLEAR région shares vs 5.08
    c1 = pd.read_csv(RAW / "clearglobal_gin_admin1.csv")
    c1 = c1[c1["language_code"] != "Unknown"].copy()
    c1["share"] = 100 * c1["proportion_value"] / c1.groupby("location_code")[
        "proportion_value"].transform("sum")
    c1["reg"] = c1["location_code"].map(CLEAR_ADM1)
    cs = c1.set_index(["reg", "language_code"])["share"]
    diffs = []
    for r in ROWS:
        code = CLEAR_CODE[r]
        if code in (None, "Unknown"):
            continue
        for j, reg in enumerate(REGIONS):
            diffs.append((abs(cs.get((reg, code), 0.0) - T508[r][j]), r, reg,
                          cs.get((reg, code), 0.0), T508[r][j]))
    diffs.sort(reverse=True)
    say(diffs[0][0] <= 3.5, "8. CLEAR's région shares vs 5.08, worst five: " + "; ".join(
        f"{r} in {reg} {c:.1f} vs {t:.1f}" for _, r, reg, c, t in diffs[:5]))

    # write gn.csv
    out = []
    for r in ROWS:
        for j, reg in enumerate(REGIONS):
            p = T508[r][j]
            if p <= 0:
                continue
            out.append(dict(geo_id=reg, geo_level="region", geo_name=reg, source_category=r,
                            pct=p, count=round(p / 100 * eff[reg]), tier="measured"))
    df = pd.DataFrame(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(df)} rows, {df['count'].sum():,} people "
          f"(Effectif {EFFECTIF_TOTAL:,}; the difference is the 0,0 cells and rounding)")

    # write gn_clear.csv (prefecture shares, Unknown kept: Manya's weight)
    c2 = c2.rename(columns={"location_code": "pref", "location_name": "pref_name",
                            "language_code": "clear_code", "language_name": "clear_name",
                            "proportion_value": "share"})
    c2[["pref", "pref_name", "clear_code", "clear_name", "share"]].to_csv(
        OUT_CLEAR, index=False, encoding="utf-8")
    print(f"wrote {OUT_CLEAR}: {len(c2)} rows")


if __name__ == "__main__":
    main()
