"""Chad RGPH2 2009, first national language spoken, national -> data/normalized/td.csv.

    python sources/td_rgph.py [--fetch]

SOURCE. INSEED, Deuxième Recensement Général de la Population et de l'Habitat (RGPH2, 2009),
*Analyse thématique: État et structures de la population* (Nov 2013, 210 pp), the volume
religiondots draws Chad's religion from (religiondots/sources/td.md: only on the retired
jdownloads store of the old inseed.td, through Wayback's 2020-06-08 capture; IREDA links a
second copy as inseedtchad.com/IMG/pdf/etat-structures-juillet-2014.pdf, also dead live).
--fetch copies religiondots' verified download (read-only); the digest pins it.

QUESTION. B18A/B18B, "langues nationales parlées", asked of everyone aged 6 and over: the
national languages a person speaks, the first and second coded (Annexe 3, pp208-210: GPLANAT
"groupe de principales premières langues nationales parlées (à partir de la variable B18A)").
The census calls B18A the "première langue nationale parlée". It is the first language the
person named, which is not defined as the mother tongue (see the Arabic note below).
French is an official language, not a "langue nationale", so it is never an answer here.

TABLES READ (1-based PDF pages):
  Tableau 5.10, pp134-135  first national language x urban/rural (and sex), % to one decimal,
                           with the universe: 1,833,267 urban, 6,255,476 rural, 8,088,816
                           total, people aged 6+ who named a first national language. DRAWN.
  Tableau 5.09, p133       2.4% of the 6+ (urban 3.3, rural 2.1) named none: the gap.
  Tableau 5.02, p126       Chadian nationals by ethnic group (grand groupe, 21 rows), counts.
                           Read for the Arabic move (countries/td.py) and check 5.
  Tableau 5.07, p131       région populations (22 régions), the placement's row margins.
  Annexes 2-3, pp206-210   which ethnic groups and languages each printed row holds.
There is no language or ethnic table by région in this volume, the *Résultats globaux
définitifs*, or the sous-préfecture volume (sources/td.md §1). The microdata (NADA catalog 26)
is a data enclave in N'Djaména.

THE COUNTS. Each language's count = urban % x 1,833,267 + rural % x 6,255,476, then scaled to
the printed total 8,088,816 by largest remainder. Using both columns halves the rounding of the
one-decimal shares; check 3 compares with total % x 8,088,816.

CHECKS (all must pass):
  1. the PDF is the pinned 210-page volume (SHA-1, size, %%EOF)
  2. Tableau 5.10 parsed off pp134-135 = the transcription (34 rows x urban/rural/total)
  3. each column sums to 100 within rounding; urban/rural-weighted shares = the printed total
     column within 0.1 pp; the counts = total % x universe within 0.1 pp
  4. Tableau 5.02 parsed = transcription; its 21 rows sum to the printed 10,666,833
  5. ethnicity against language: the southern groups' languages match their ethnic shares
     (Sara group 26.6% vs its four languages); prints the Arabic gap the move closes
  6. Tableau 5.07's 22 région populations sum to 10,941,682 and match religiondots' td.csv
"""
import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

RAW = HERE / "data" / "raw" / "td"
PDF = RAW / "rgph2_etat_structures.pdf"
RD_PDF = RD / "data" / "raw" / "td" / "rgph2_etat_structures.pdf"
SHA1 = "0967fb250dd4c57aa50c3b8b6b7c5fe280bd568e"
OUT = HERE / "data" / "normalized" / "td.csv"

URBAN, RURAL, TOTAL = 1_833_267, 6_255_476, 8_088_816

# Tableau 5.10 as printed: label -> (urban %, rural %, total %)
T510 = {
    "Arabe local": (31.9, 18.8, 21.7),
    "Sara": (23.5, 23.9, 23.8),
    "Gorane": (7.7, 6.5, 6.8),
    "Kanembou": (4.7, 6.1, 5.8),
    "Maba/Ouaddaï": (2.4, 4.0, 3.6),
    "Moundang": (2.2, 2.6, 2.5),
    "Mousseye": (1.1, 3.2, 2.7),
    "Boulala": (2.0, 1.7, 1.8),
    "Zaghawa/Béri/Bideyat": (2.7, 2.4, 2.5),
    "Marba": (0.9, 2.2, 1.9),
    "Massa": (1.5, 1.6, 1.5),
    "Peul/Foulfouldé/Bodoré": (1.3, 2.1, 1.9),
    "Barma/Baguirmi": (0.7, 0.5, 0.5),
    "Massalit": (0.5, 1.8, 1.5),
    "Mimi": (0.2, 0.4, 0.4),
    "Rounga": (0.1, 0.2, 0.2),
    "Tama": (0.2, 1.3, 1.0),
    "Dadjo": (1.3, 1.2, 1.2),
    "Moubi": (0.5, 0.7, 0.6),
    "Mesmé": (0.2, 0.0, 0.1),
    "Gabri": (0.5, 1.1, 0.9),
    "Kabalaye": (0.5, 0.1, 0.2),
    "Kéra": (0.4, 0.7, 0.6),
    "Kim": (0.5, 0.2, 0.2),
    "Lamé/Pévé": (0.0, 0.7, 0.5),
    "Lélé": (0.9, 1.0, 1.0),
    "Nangtchéré": (0.6, 1.0, 1.0),
    "Toupouri": (1.2, 1.5, 1.4),
    "Karo/Kado": (0.7, 0.9, 0.8),
    "Daye": (0.7, 0.7, 0.7),
    "Mboum": (0.3, 0.4, 0.4),
    "Sara Kaba": (1.2, 1.8, 1.7),
    "Toumak/Ndom": (0.1, 0.4, 0.4),
    "Autres 1ere langues nationales parlées": (6.7, 8.3, 8.0),
}

# Tableau 5.02, Chadian nationals by grand groupe ethnique, Ensemble column
T502 = {
    "Gorane": 739_819,
    "Arabe": 1_378_777,
    "Baguirmi/Barma et autres": 139_799,
    "Kanembou/Bornou/Boudouma": 902_872,
    "Boulala/Médégo/Kouka": 395_238,
    "Ouaddaï/Maba/Massalit/Mimi": 765_942,
    "Zaghawa (Bideyat/Kobé)": 252_921,
    "Dadjo/Kibet/Mouro et autres": 276_827,
    "Bidio/Migami/Kinga/dangléat et autres": 395_034,
    "Moundang": 270_822,
    "Massa/Mousseye/Mousgoume": 515_685,
    "Toupouri/Kéra": 215_466,
    "Sara (Ngambaye/Sara Madjingaye/Mbaye et autres)": 2_836_371,
    "Peul/Foulbé/Bodoré": 223_782,
    "Tama/Assongori/Mararit": 175_950,
    "Gabri/Kabalaye/Nangtchéré/Soumraye et autres": 259_309,
    "Marba/Lélé/Mesmé": 322_021,
    "Mesmedjé/Massalat/Kadjaksé": 108_023,
    "Karo/Zimé/Pévé": 146_464,
    "Autres ethnies tchadiennes (Achit/Banda/Kim et autres)": 275_725,
    "Autres ethnies d'origine étrangère": 69_989,
}
T502_TOTAL = 10_666_833

# Tableau 5.07, région -> Effectif
T507 = {
    "Batha": 488458, "Borkou": 93584, "Chari Baguirmi": 578425, "Guéra": 538359,
    "Hadjer Lamis": 566858, "Kanem": 333387, "Lac": 433790, "Logone Occidental": 689044,
    "Logone Oriental": 779339, "Mandoul": 628065, "Mayo Kebbi Est": 774782,
    "Mayo Kebbi Ouest": 564470, "Moyen Chari": 588008, "Ouaddaï": 721166, "Salamat": 302301,
    "Tandjilé": 661906, "Wadi Fira": 508383, "N'Djaména": 951418, "Barh El Gazal": 257267,
    "Ennedi": 167919, "Sila": 293450, "Tibesti": 21303,
}
T507_TOTAL = 10_941_682


def fold(s):
    return re.sub(r"/ ", "/", re.sub(r"\s+", " ", s.replace("’", "'"))).strip()


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    if not RD_PDF.exists():
        raise SystemExit(f"{RD_PDF} missing: run religiondots' sources/td.py --fetch, or fetch "
                         "the Wayback capture named in its sources/td.md")
    shutil.copyfile(RD_PDF, PDF)
    print(f"copied {RD_PDF.name} from religiondots ({PDF.stat().st_size:,} bytes)")


def nums(lines):
    out = []
    for ln in lines:
        t = ln.strip().replace(" ", " ")
        if re.fullmatch(r"[\d ]+(,\d+)?", t) and t:
            out.append(float(t.replace(" ", "").replace(",", ".")))
    return out


def parse_510(doc):
    lines = [l.strip() for l in (doc[133].get_text() + doc[134].get_text()).splitlines()]
    i = lines.index("Arabe local")
    rows, cur = {}, None
    vals = []
    for ln in lines[i:]:
        if ln.startswith("Total"):
            break
        if re.fullmatch(r"[\d ]+(,\d+)?", ln):
            vals.append(float(ln.replace(" ", "").replace(",", ".")))
            if len(vals) == 8:
                rows[cur] = (vals[0], vals[1], vals[2])
                vals = []
        elif ln and not re.fullmatch(r"\d+", ln):
            cur = fold(ln)
            vals = []
    return rows


def parse_502(doc):
    lines = [l.strip() for l in doc[125].get_text().splitlines()]
    i = lines.index("Gorane")
    rows, label, vals = {}, [], []
    for ln in lines[i:]:
        if ln.startswith("Total"):
            break
        if re.fullmatch(r"[\d ]+(,\d+)?", ln):
            vals.append(ln)
            if len(vals) == 6:
                rows[fold(" ".join(label))] = int(vals[2].replace(" ", ""))
                label, vals = [], []
        elif ln:
            label.append(ln)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch or not PDF.exists():
        fetch()

    import fitz
    data = PDF.read_bytes()
    sha = hashlib.sha1(data).hexdigest()
    doc = fitz.open(PDF)
    assert sha == SHA1, f"SHA-1 {sha}, expected {SHA1}"
    assert len(doc) == 210 and b"%%EOF" in data[-64:], "not the 210-page volume, or truncated"
    print(f"1. {PDF.name}: 210 pages, {len(data):,} bytes, SHA-1 pinned, %%EOF  ok")

    p510 = parse_510(doc)
    want = {fold(k): v for k, v in T510.items()}
    assert p510 == want, f"Tableau 5.10 parse != transcription: " \
        f"{set(p510.items()) ^ set(want.items())}"
    print(f"2. Tableau 5.10: {len(p510)} rows parsed = transcription  ok")

    u = sum(v[0] for v in T510.values())
    r = sum(v[1] for v in T510.values())
    t = sum(v[2] for v in T510.values())
    print(f"3. column sums: urban {u:.1f}, rural {r:.1f}, total {t:.1f}")
    assert all(abs(s - 100) <= 0.6 for s in (u, r, t)), "a column is off 100 by more than rounding"
    raw = {k: v[0] / 100 * URBAN + v[1] / 100 * RURAL for k, v in T510.items()}
    tot = sum(raw.values())
    worst = 0.0
    for k, v in T510.items():
        mix = raw[k] / (URBAN + RURAL) * 100
        worst = max(worst, abs(mix - v[2]))
        assert abs(mix - v[2]) <= 0.1, f"{k}: urban/rural mix {mix:.2f} vs printed total {v[2]}"
    print(f"   urban/rural mix against the printed total column: worst {worst:.3f} pp  ok")
    # scale to the printed universe, largest remainder
    exact = {k: x * TOTAL / tot for k, x in raw.items()}
    cnt = {k: int(x) for k, x in exact.items()}
    short = TOTAL - sum(cnt.values())
    for k in sorted(exact, key=lambda k: exact[k] - cnt[k], reverse=True)[:short]:
        cnt[k] += 1
    assert sum(cnt.values()) == TOTAL
    dev = max(abs(cnt[k] / TOTAL * 100 - v[2]) for k, v in T510.items())
    assert dev <= 0.1, dev
    print(f"   counts sum to {TOTAL:,}; largest gap to total % x universe {dev:.3f} pp  ok "
          f"(the columns sum to {u:.1f}/{r:.1f}, so each share is rescaled by ~{100 / t:.4f})")

    p502 = parse_502(doc)
    want502 = {fold(k): v for k, v in T502.items()}
    got502 = {re.sub(r" e t ", " et ", k): v for k, v in p502.items()}
    assert got502 == want502, f"Tableau 5.02 parse != transcription: " \
        f"{set(got502.items()) ^ set(want502.items())}"
    s502 = sum(T502.values())
    assert abs(s502 - T502_TOTAL) <= 25, s502
    print(f"4. Tableau 5.02: 21 rows parsed = transcription; sum {s502:,} vs printed "
          f"{T502_TOTAL:,} ({s502 - T502_TOTAL:+d})  ok")

    sara_e = T502["Sara (Ngambaye/Sara Madjingaye/Mbaye et autres)"] / T502_TOTAL * 100
    sara_l = sum(cnt[k] for k in ("Sara", "Sara Kaba", "Daye", "Mboum")) / TOTAL * 100
    ar_e = T502["Arabe"] / T502_TOTAL * 100
    ar_l = cnt["Arabe local"] / TOTAL * 100
    print(f"5. Sara: ethnic {sara_e:.1f}% vs Sara + Sara Kaba + Daye + Mboum {sara_l:.1f}%; "
          f"Arab: ethnic {ar_e:.1f}% vs Arabe local first {ar_l:.1f}% (gap {ar_l - ar_e:.1f} pp)")
    assert abs(sara_e - sara_l) < 1.0

    s507 = sum(T507.values())
    assert s507 == T507_TOTAL, s507
    rd = pd.read_csv(RD / "data" / "normalized" / "td.csv")
    rdpop = rd.groupby("geo_id")["count"].sum()
    bad = {k: (v, rdpop.get(k)) for k, v in T507.items() if rdpop.get(k) != v}
    assert not bad, f"région populations differ from religiondots' td.csv: {bad}"
    print(f"6. Tableau 5.07: 22 régions sum to {s507:,}, = religiondots' td.csv  ok")

    rows = [dict(geo_id="TD", geo_level="country", geo_name="Chad", source_category="Total",
                 count=TOTAL, tier="measured", year=2009, source_id="td_rgph2_2009_t510",
                 note="universe: aged 6+ naming a first national language (Tableau 5.10 "
                      "Effectif); not a language")]
    for k, v in T510.items():
        rows.append(dict(geo_id="TD", geo_level="country", geo_name="Chad", source_category=k,
                         count=cnt[k], tier="measured", year=2009,
                         source_id="td_rgph2_2009_t510",
                         note=f"Tableau 5.10 urban {v[0]}% rural {v[1]}% total {v[2]}%; "
                              "urban x 1,833,267 + rural x 6,255,476, scaled to 8,088,816"))
    for k, v in T502.items():
        rows.append(dict(geo_id="TD", geo_level="country_ethnic", geo_name="Chad",
                         source_category=k, count=v, tier="measured", year=2009,
                         source_id="td_rgph2_2009_t502",
                         note="Tableau 5.02, Chadian nationals by grand groupe ethnique; "
                              "not drawn, read for the Arabic move"))
    for k, v in T507.items():
        rows.append(dict(geo_id=k, geo_level="region", geo_name=k, source_category="Population",
                         count=v, tier="measured", year=2009, source_id="td_rgph2_2009_t507",
                         note="Tableau 5.07 Effectif; placement row margin, not a language"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT.relative_to(HERE)}: {len(T510)} languages, {len(T502)} ethnic groups, "
          f"{len(T507)} régions")


if __name__ == "__main__":
    main()
