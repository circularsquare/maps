"""Madagascar RGPH-3 2018, ability to speak Malagasy, French, English and other languages, per
region -> data/normalized/mg.csv.

    python sources/mg_rgph.py [--fetch]

SOURCE. INSTAT, *Troisième Recensement Général de la Population et de l'Habitation (RGPH-3),
Thème: Compétences linguistiques et scolarisation à Madagascar* (dated February 2021 in the file
name, October 2021 on its pages), PDF, 155 pages, no login:
https://www.instat.mg/documents/upload/main/INSTAT-RGPH3_CompetencesLinguistiquesetScolarisation_Fev-2021.pdf
(11,101,312 bytes, digest pinned). instat.mg sometimes answers with a Cloudflare challenge; a
browser User-Agent got the file on 2026-10-05.

QUESTION. Whether each person aged 3 or over can speak Malagasy, French, English, "other
languages"; four yes/no items, so several allowed. The report calls the last "autres langues
étrangères" (p.17). Tableau 2.2 (PDF p.48): per region, the population aged 3 and over and the
percentage able to speak each of the four, one decimal. Percentages only.

The total population per region (all ages, all households) is RGPH-3 Tome 1, Tableau 6, as
religiondots transcribed and checked it into `religiondots/data/geo/mg/mg_lookup.csv` (read only;
its sum, 25,674,196, is the census's published total).

OUTPUT rows per region (geo_id = religiondots' MG11..MG72):
  Population            all ages, Tableau 6
  Population 3+         Tableau 2.2
  Malagasy, Français, Anglais, Autres langues   pct and count = pct * Population 3+ / 100

CHECKS (all must pass):
  1. the PDF is the pinned file (digest)
  2. 22 regions, every one joining one-to-one to religiondots' lookup by name
  3. the regions' Population 3+ adds to the printed national 23,507,970 exactly; Tableau 2.1's
     urban + rural and male + female also add to it
  4. the national printed shares (99.9, 23.6, 8.2, 0.6) follow from the regions weighted by
     Population 3+, within 0.06
  5. Population 3+ over Tableau 6's population per region lies within 0.85-0.95 (under-3s are
     about 8-10% of each region, plus collective households): a wrong-twin guard on the name
     join, as the populations come from two tables
"""
import argparse
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

RAW = HERE / "data" / "raw" / "mg"
NORM = HERE / "data" / "normalized" / "mg.csv"
NAME = "INSTAT-RGPH3_CompetencesLinguistiquesetScolarisation_Fev-2021.pdf"
URL = "https://www.instat.mg/documents/upload/main/" + NAME
SHA = "a079629004840c35b9a1140e83ac20effe449994cb03eecf347c77cc662ee06e"
LANGS = ("Malagasy", "Français", "Anglais", "Autres langues")
NATIONAL_3P = 23_507_970
NATIONAL_PCT = (99.9, 23.6, 8.2, 0.6)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0 Safari/537.36")


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        (RAW / NAME).write_bytes(r.read())


def num(s):
    return float(s.replace(" ", "").replace("\xa0", "").replace(" ", "").replace(",", "."))


def key(name):
    return re.sub(r"[^a-z]", "", name.lower())


def read():
    body = (RAW / NAME).read_bytes()
    if hashlib.sha256(body).hexdigest() != SHA:
        raise SystemExit(f"{NAME} is not the pinned file; check it, then update SHA")   # check 1
    import fitz
    doc = fitz.open(RAW / NAME)
    t21 = [ln.strip() for ln in doc[46].get_text().splitlines() if ln.strip()]
    t22 = [ln.strip() for ln in doc[47].get_text().splitlines() if ln.strip()]
    if not any(ln.startswith("Tableau 2.2.") for ln in t22):
        raise SystemExit("PDF p.48 is not Tableau 2.2")
    i = t22.index("Région") + 1
    rows = []
    while True:
        name, vals = t22[i], t22[i + 1:i + 6]
        rows.append([name] + [num(v) for v in vals])
        i += 6
        if name == "MADAGASCAR":
            break
    t22 = pd.DataFrame(rows, columns=["geo_name", "pop3"] + list(LANGS))
    # Tableau 2.1: urban, rural, male, female rows (population 3+ is the number after the label)
    t21v = {lab: num(t21[t21.index(lab) + 1]) for lab in ("Urbain", "Rural", "Masculin", "Féminin")}
    return t22, t21v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch or not (RAW / NAME).exists():
        fetch()
    t22, t21v = read()
    nat = t22[t22["geo_name"] == "MADAGASCAR"].iloc[0]
    reg = t22[t22["geo_name"] != "MADAGASCAR"].copy()

    # ---- check 2: the join ----
    lut = pd.read_csv(RD_GEO / "mg" / "mg_lookup.csv", dtype={"pop": int})
    lk = {key(n): (g, p) for g, n, p in zip(lut["geo_id"], lut["name"], lut["pop"])}
    reg["k"] = reg["geo_name"].map(key)
    if len(reg) != 22 or reg["k"].nunique() != 22 or set(reg["k"]) != set(lk):
        raise SystemExit(f"regions do not join one-to-one: {sorted(set(reg['k']) ^ set(lk))}")
    reg["geo_id"] = reg["k"].map(lambda k: lk[k][0])
    reg["pop"] = reg["k"].map(lambda k: lk[k][1])
    print(f"check 2: 22 regions join one-to-one to religiondots' lookup "
          f"(population {reg['pop'].sum():,})")

    # ---- check 3: population 3+ ----
    s = reg["pop3"].sum()
    if not (s == nat["pop3"] == NATIONAL_3P == t21v["Urbain"] + t21v["Rural"]
            == t21v["Masculin"] + t21v["Féminin"]):
        raise SystemExit(f"population 3+ does not close: regions {s:,.0f}, printed "
                         f"{nat['pop3']:,.0f}, Tableau 2.1 {t21v}")
    print(f"check 3: regions add to {s:,.0f} = national = urban + rural = male + female")

    # ---- check 4: national shares ----
    for lang, want in zip(LANGS, NATIONAL_PCT):
        got = (reg[lang] * reg["pop3"]).sum() / s
        print(f"check 4: {lang:15s} regions give {got:6.3f}, printed {nat[lang]} ({want})")
        if nat[lang] != want or abs(got - want) > 0.06:
            raise SystemExit(f"{lang}: national share does not follow from the regions")

    # ---- check 5: wrong-twin guard ----
    reg["r"] = reg["pop3"] / reg["pop"]
    print(f"check 5: population 3+ / Tableau 6 population, {reg['r'].min():.3f} "
          f"({reg.loc[reg['r'].idxmin(), 'geo_name']}) to {reg['r'].max():.3f} "
          f"({reg.loc[reg['r'].idxmax(), 'geo_name']}); national {s / reg['pop'].sum():.3f}")
    if not reg["r"].between(0.85, 0.95).all():
        raise SystemExit("a region's 3+ population is out of line with its total")

    out = []
    for _, r in reg.iterrows():
        base = dict(geo_id=r["geo_id"], geo_name=r["geo_name"], geo_level="region")
        out.append({**base, "source_category": "Population", "pct": None, "count": r["pop"]})
        out.append({**base, "source_category": "Population 3+", "pct": None, "count": r["pop3"]})
        for lang in LANGS:
            out.append({**base, "source_category": lang, "pct": r[lang],
                        "count": r[lang] * r["pop3"] / 100})
    NORM.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(NORM, index=False)
    print(f"wrote {NORM} ({len(out)} rows)")
    for lang in LANGS:
        print(f"  {lang:15s} {(reg[lang] * reg['pop3']).sum() / 100:>14,.0f} able to speak it (3+)")


if __name__ == "__main__":
    main()
