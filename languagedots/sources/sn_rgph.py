"""Senegal RGPH-5 2023, main language most often spoken, by région -> data/normalized/sn.csv.

    python sources/sn_rgph.py [--fetch]

SOURCE. ANSD (Agence nationale de la Statistique et de la Démographie), RGPH-5 2023: the
fourteen regional reports ("RGPH-5 : Rapport de présentation des résultats, Région de ...",
April 2026) and the national final report's chapter 1 ("Etat et structure, urbanisation et
caractéristiques socioculturelles de la population"). All listed at
https://www.ansd.sn/rapports/rgph-5-2023, no login. ANSD's TLS chain is broken (curl -k, or
verify=False here); the digests below are what makes the files trustworthy.

QUESTION. B18 "première langue la plus souvent parlée" (the language the person speaks most
often), residents aged 3 and over, one answer; B19 asked the second language and is not
tabulated by région. The tables call it "principale langue couramment parlée". Chapter 1's
Tableau I-1 reports no missing values for B18.

TABLES READ (1-based PDF pages; the table number differs between reports):
  each regional report, Tableau II-11/II-12/II-13 (Diourbel misnumbers it III-12): the région's
      residents aged 3+ by principal language x sex, counts and %, 23 rows + Total.
  chapter 1, Tableau I-32, pp57-58: the same table nationally (the second table).

CHECKS (all must pass):
  1. every PDF is the pinned file (size, sha256, %%EOF)
  2. each regional table: the 23 expected labels, once each; 6 cells per row; masculin +
     féminin = ensemble within 1 on every row; the rows sum to the printed Total within 2
  3. Tableau I-32's own rows sum to its printed total
  4. the 14 régions fall short of Tableau I-32 by exactly the pinned 492,752 (3.0%); per
     language the ratio is printed (0.94 sign language to 0.99 French, the big languages
     0.968-0.976). Not explained anywhere in the reports; the regional counts are drawn as
     printed and the difference goes in `gap`.

TRAPS: the reports differ in layout (labels wrapped over two lines, a repeated column head at a
page break, Ziguinchor's SPSS variable name in the head); Saint-Louis prints a second "Total"
row that repeats the sign-language row (ignored, the first Total closes); Diourbel numbers the
table III-12.
"""
import argparse
import hashlib
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "sn"
OUT = HERE / "data" / "normalized" / "sn.csv"
BASE = "https://www.ansd.sn/sites/default/files/recensements/rapport/"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

# région -> (local file, URL tail, 1-based first and last page of the language table)
REPORTS = {
    "Dakar": ("rr_dakar.pdf", "RGPH-5_DAKAR_Avril%202026.pdf", 26, 26),
    "Diourbel": ("rr_diourbel.pdf", "RGPH-5%20RR_DIOURBEL_27112025_Version%20Formate%CC%81%20ok_Avril%202026.pdf", 31, 31),
    "Fatick": ("rr_fatick.pdf", "RGPH-5%20RR_FATICK_Avril%202026_0.pdf", 34, 35),
    "Kaffrine": ("rr_kaffrine.pdf", "RGPH-5%20RR_KAFFRINE_Avril%202026.pdf", 37, 38),
    "Kaolack": ("rr_kaolack.pdf", "RGPH-5%20RR_KAOLACK_27112025_Version%20Formate%CC%81%20ok_Avril%202026_0.pdf", 41, 42),
    "Kédougou": ("rr_kedougou.pdf", "RGPH-5%20RR_KEDOUGOU_Avril%202026.pdf", 38, 39),
    "Kolda": ("rr_kolda.pdf", "RGPH-5%20RR_KOLDA_27112025_Version%20Formate%CC%81%20ok_Avril%202026_0.pdf", 38, 39),
    "Louga": ("rr_louga.pdf", "RGPH-5%20RR_LOUGA_27112025_Version%20Formate%CC%81%20ok_Avril%202026.pdf", 41, 42),
    "Matam": ("rr_matam.pdf", "RGPH-5%20RR_MATAM_27112025_Version%20Formate%CC%81%20ok_Avril%202026_0.pdf", 38, 39),
    "Saint-Louis": ("rr_saint_louis.pdf", "RGPH-5%20RR_SAINT_LOUIS_Avril%202026.pdf", 28, 29),
    "Sédhiou": ("rr_sedhiou.pdf", "RGPH-5%20RR_SEDHIOU_27112025_Version%20Formate%CC%81%20ok_Avril%202026.pdf", 32, 33),
    "Tambacounda": ("rr_tambacounda.pdf", "RGPH-5%20RR_TAMBACOUNDA_Avril%202026_0.pdf", 35, 36),
    "Thiès": ("rr_thies.pdf", "RGPH-5%20RR_THIES_Avril%202026.pdf", 26, 27),
    "Ziguinchor": ("rr_ziguinchor.pdf", "RGPH-5_ZIGUINCHOR_Avril%202026_0.pdf", 33, 34),
}
NATIONAL = ("rgph5_ch1.pdf", "Chapitre-1_ETAT-STRUCTURE-POPULATION-Rapport-def-RGPH-5.pdf", 57, 58)

PINS = {  # file -> (bytes, sha256), downloaded 2026-10-05
    "rr_dakar.pdf": (13616720, "e027a15cc6d29786fea36de5a4880043c5b0f2d7e3c2132878ec0d8bdfa0bc89"),
    "rr_diourbel.pdf": (15068454, "7a79677cc7438934eb2b04e4e740109064b3d4f26660d5c65e007c8a66f69ef6"),
    "rr_fatick.pdf": (17326890, "ea6299540060cae9f69df7d079e4c9f40c8c1fb5c88b2cf40402dcc45985f93b"),
    "rr_kaffrine.pdf": (14232161, "c1cc6366ed93e32c1b08bb223f9c2980935c7ded11701640208cd7e45804101d"),
    "rr_kaolack.pdf": (12309217, "a33b96596c62bef85242d978b0d4051bf52a0dc313367c94ce05975e4f063708"),
    "rr_kedougou.pdf": (14630701, "3f1ed03faa29f1d51e66df6aa7e436e0577b0e7c579ae9e3e4ce33d9f6d5b259"),
    "rr_kolda.pdf": (12694365, "ab6235c1399fa33d1a97066be8f6ff59d3699ce1b3677fd435571a295e413366"),
    "rr_louga.pdf": (15374077, "d25a403566b9840d57e3120127e1719819c75cf81ca6a264a8ded05c7be56a9b"),
    "rr_matam.pdf": (13033059, "653bcf8176a2e7a66aab019223cf220bd23ee92d9c45ca535d4291eaaec5d20d"),
    "rr_saint_louis.pdf": (12142290, "d529cc53695f3b6c97a4c75933d67cfa0af1688ebd16c1570bb753e072ecb0bb"),
    "rr_sedhiou.pdf": (13152192, "5e9e1a1ef06e8aaebd511f1d5c4b7c8e39ac10d8bf9041be75e823830a36bf45"),
    "rr_tambacounda.pdf": (14446727, "f4a02f4e7ad204a18a35d084f7acf43769a0d4ada28651d2148a4e8f5ce068ee"),
    "rr_thies.pdf": (12432668, "b6112b28b181d1306d5852244c004ebddcc68d154e2837e6961b6b1cc54ea896"),
    "rr_ziguinchor.pdf": (24959248, "ce0bf1b80a6f64f30bdd5011ef42852ddcf0037a7c0b4b577c381738b3e3629e"),
    "rgph5_ch1.pdf": (2773316, "8806140091f59bd623da9d49fb7d6f873c97686b62a2192fa91957b26b957f25"),
}
# The fourteen regional tables sum to this many fewer people than Tableau I-32 (3.0%, about the
# same share of every language); measured 2026-10-05, pinned so a re-issued report shows up.
NATIONAL_EXCESS = 492_752

# Tableau I-32's labels, the source_category spelling; regional spellings fold onto them.
LABELS = [
    "Wolof", "Pulaar", "Sereer", "Joola", "Màndienka", "Sóninke", "Hasaniya (Maure)", "Balante",
    "Mànkaañ", "Mànjaku", "Mënik", "Oniyan", "Guñuun", "Kanjad", "Jalunga", "Bayot", "Womey",
    "Tourka (Sénégal)", "Langage des signes (Sourd-Muet)", "Langues étrangères",
    "Autres langues africaines", "Français", "Autres langues étrangères non africaines",
]


def key(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


KEYS = {key(lab): lab for lab in LABELS}
KEYS.update({
    "balant": "Balante",
    "tourka": "Tourka (Sénégal)",
    "langueetrangere": "Langues étrangères",
    "languesetrangere": "Langues étrangères",
    "languesetrangeres": "Langues étrangères",
    "langueetrangeres": "Langues étrangères",
    "hasaniyamaure": "Hasaniya (Maure)",
})


# column and running heads a table repeats at a page break
HEADER_WORDS = {
    "principale", "principales", "langue", "couramment", "parlee", "parlees",
    "principaleslanguescouramment",
    "principalelangue", "courammentparlee", "principalelanguecourammentparlee",
    "principaleslanguescourammentparlees", "principaleslanguescourammentparlee", "sexe",
    "masculin", "feminin", "ensemble", "effectif", "proportion", "proportio", "n",
    "rapportdefinitif", "ansd",
}


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, tail, *_ in list(REPORTS.values()) + [NATIONAL]:
        out = RAW / name
        if out.exists() and out.stat().st_size > 1_000_000:
            continue
        print("GET", BASE + tail)
        subprocess.run(["curl", "-k", "-sS", "-L", "-A", UA, "-o", str(out), BASE + tail], check=True)


def check_file(name):
    p = RAW / name
    data = p.read_bytes()
    if b"%%EOF" not in data[-2048:]:
        raise SystemExit(f"{name}: no %%EOF trailer, truncated")
    got = (len(data), hashlib.sha256(data).hexdigest())
    want = PINS.get(name)
    if want is None:
        print(f"  UNPINNED {name}: {got}")
    elif got != want:
        raise SystemExit(f"{name}: {got} != pinned {want}")


def cells(line):
    """Split one text line into cells: digit groups of a count are rejoined ('1 228 269'); a
    percentage ('81,3') ends a cell; '-' is zero."""
    out = []
    for tok in line.split():
        if out and re.fullmatch(r"\d{3}", tok) and re.fullmatch(r"\d{1,3}( \d{3})*", out[-1]):
            out[-1] += " " + tok
        else:
            out.append(tok)
    return out


def num(c):
    if c in ("-", "–"):
        return 0.0
    c = c.replace(" ", "").replace(" ", "").replace("\xa0", "")
    if c.startswith(","):
        c = "0" + c
    return float(c.replace(",", "."))


def parse_table(name, p0, p1, start_label):
    import fitz
    doc = fitz.open(RAW / name)
    lines = []
    for i in range(p0, p1 + 1):
        # Ziguinchor's table keeps SPSS's variable name and column caption in its head
        page = [ln.replace("1ER_LANGUE_RECOD", "").replace("Effectif Nb.colonnes", "").strip()
                for ln in doc[i - 1].get_text().splitlines()]
        # the running head's page number sits in the first few lines; "%" is a column head
        page = [ln for j, ln in enumerate(page)
                if not (j < 5 and re.fullmatch(r"\d{1,3}", ln)) and ln not in ("%", "(%)")]
        lines += page
    # from the first data row to the Total row's "Source" line
    k = next(i for i, ln in enumerate(lines) if key(ln).startswith(key(start_label)))
    rows, cur, vals = [], None, []
    total = None
    for ln in lines[k:]:
        if not ln:
            continue
        if ln.startswith("Source"):
            break
        if re.search(r"[A-Za-zÀ-ÿ]", ln) and not re.fullmatch(r"[\d\s,.\-–]+", ln):
            kk = key(ln)
            if kk in HEADER_WORDS or kk.startswith("region") or kk.startswith("theme") \
                    or kk.startswith("rgph"):
                continue
            if cur is not None and not vals:      # a label wrapped onto a second line
                cur += " " + ln
                continue
            if cur is not None:
                rows.append((cur, vals))
            cur, vals = ln, []
            continue
        if cur is None:
            continue
        vals += cells(ln)
    if cur is not None:
        rows.append((cur, vals))
    out = {}
    for lab, v in rows:
        if key(lab) == "total":
            if total is not None:
                # Saint-Louis prints a second "Total" row repeating the sign-language row
                print(f"    {name}: a second Total row {v} ignored")
                continue
            if len(v) != 6:
                raise SystemExit(f"{name}: Total has {len(v)} cells: {v}")
            total = [num(x) for x in v]
            continue
        canon = KEYS.get(key(lab))
        if canon is None:
            raise SystemExit(f"{name}: unknown label {lab!r} ({v})")
        if canon in out:
            raise SystemExit(f"{name}: {canon} twice")
        if len(v) != 6:
            raise SystemExit(f"{name}: {lab!r} has {len(v)} cells: {v}")
        m, _, f, _, e, pct = (num(x) for x in v)
        if abs(m + f - e) > 1:
            raise SystemExit(f"{name}: {lab}: {m} + {f} != {e}")
        out[canon] = (int(e), pct)
    if sorted(out) != sorted(LABELS):
        raise SystemExit(f"{name}: labels {sorted(set(LABELS) ^ set(out))} missing or extra")
    if total is None:
        raise SystemExit(f"{name}: no Total row")
    s = sum(e for e, _ in out.values())
    if abs(s - total[4]) > 2:
        raise SystemExit(f"{name}: rows sum to {s:,}, Total prints {total[4]:,.0f}")
    return out, int(total[4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    for name, *_ in list(REPORTS.values()) + [NATIONAL]:
        check_file(name)

    recs = []
    reg_tot = {}
    for reg, (name, _, p0, p1) in REPORTS.items():
        first = "Langage"
        t, tot = parse_table(name, p0, p1, first)
        reg_tot[reg] = tot
        for lab in LABELS:
            e, pct = t[lab]
            recs.append(dict(geo_id=reg, geo_level="region", geo_name=reg, source_category=lab,
                             pct=pct, count=e, tier="measured"))
        print(f"  {reg:12s} {tot:>10,}  {len(t)} rows ok")

    nat, nat_tot = parse_table(NATIONAL[0], NATIONAL[2], NATIONAL[3], "Wolof")
    df = pd.DataFrame(recs)
    s = df.groupby("source_category")["count"].sum()
    print(f"  régions {s.sum():,} against Tableau I-32's total {nat_tot:,}")
    rows_sum = sum(e for e, _ in nat.values())
    print(f"  Tableau I-32's rows sum to {rows_sum:,}, its Total prints {nat_tot:,}")
    if abs(rows_sum - nat_tot) > 3:
        raise SystemExit("Tableau I-32's rows do not sum to its total")
    if nat_tot - int(s.sum()) != NATIONAL_EXCESS:
        raise SystemExit(f"national - régions = {nat_tot - int(s.sum()):,}, pinned {NATIONAL_EXCESS:,}")
    for lab in LABELS:
        d = int(s[lab]) - nat[lab][0]
        print(f"    {lab:42s} régions {int(s[lab]):>10,}  national {nat[lab][0]:>10,}  "
              f"{d:+9,}  ratio {s[lab] / nat[lab][0]:.3f}")
    for lab in LABELS:
        df.loc[len(df)] = dict(geo_id="SN", geo_level="national", geo_name="Sénégal",
                               source_category=lab, pct=nat[lab][1], count=nat[lab][0],
                               tier="measured")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"  wrote {OUT} ({len(df)} rows)")


if __name__ == "__main__":
    main()
