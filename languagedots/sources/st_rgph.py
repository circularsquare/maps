"""São Tomé and Príncipe, IV RGPH 2012 (INE): Quadro 10, "População residente segundo sexo e
grupos etários por língua falada", from the seven district reports and the national one.

    python sources/st_rgph.py --fetch     the eight PDFs (~36 MB) into data/raw/st/
    python sources/st_rgph.py             normalise -> data/normalized/st.csv

THE QUESTION is "which languages do you speak", several allowed, asked of everyone aged one and
over. Quadro 10 prints, for each of eight languages, Total / Sim / Não: Português, Fôrro,
Angolar, Lunguié, Cabo verdiano, Francês, Inglês, and `Outra(s) língua(s) (inclusive sinais)`.
Its universe is the population aged 1+: 173,015 nationally against 178,739 people counted, so
5,724 infants under one are not in the table.

THE SOURCE FILES are the same eight PDFs religiondots fetched for its religion table (Quadro 8;
`../religiondots/sources/st.py` documents the open autoindex on www.ine.st). This script reads
data/raw/st/ when it holds them and otherwise reads religiondots' copies, read-only, so the
36 MB is not duplicated unless --fetch is run.

Writes one row per (district, label): `count` is the Sim figure (people who named it), and
`População 1+` is the table's Total, the denominator countries/st.py shares people across.

CHECKS (the script stops unless all hold):
  1. Per district and nationally, every language's Total is the same figure and Sim + Não equals
     it, which pins the parse to the right cells.
  2. The seven districts sum to the national report's Quadro 10 on every Sim and on the Total.
  3. Each district's 1+ population is below its Quadro 8 (all ages) figure from religiondots'
     parse, by 2.5-4.5%: the under-ones (printed, and asserted in that band).
  4. Each district's Portuguese Sim is at most its Total, and the sum of Sims is at least the
     Total less the people who said no to Portuguese (everyone named something, near enough).
"""
import csv
import os
import re
import sys
import urllib.parse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "st"
RD_RAW = HERE.parent / "religiondots" / "data" / "raw" / "st"      # read-only
OUT = HERE / "data" / "normalized" / "st.csv"

BASE = "https://www.ine.st"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126 Safari/537.36"}
DIR_2012 = ("/phocadownload/userupload/Documentos/Recenseamentos/2012/"
            "Dados Distritais e Nacional Recenseamento 2012/")

# COD-AB pcode (religiondots' placement units) -> (file, name, remote file)
DISTRICTS = {
    "ST21": ("d_agua_grande.pdf", "Água-Grande", "Resultado_Distrital__ÁGUA-GRANDE.pdf"),
    "ST22": ("d_cantagalo.pdf", "Cantagalo", "Resultado Distrital_CANTAGALO.pdf"),
    "ST23": ("d_caue.pdf", "Caué", "Resultado_Distrital_CAUÉ.pdf"),
    "ST24": ("d_lemba.pdf", "Lembá", "Resultado_Distrital_LEMBÁ.pdf"),
    "ST25": ("d_lobata.pdf", "Lobata", "Resultado Distrital_LOBATA.pdf"),
    "ST26": ("d_me_zochi.pdf", "Mé-Zóchi", "Resultado_Distrital_MÉ-ZÓCHI.pdf"),
    "ST11": ("d_principe.pdf", "Região Autónoma do Príncipe",
             "Resultado Distrital_REGIÃO AUTÓNOMA DO PRÍNCIPE.pdf"),
}
NATIONAL = ("nacional2012.pdf", "Resultados Nacionais do IV RGPH 2012.pdf")

# Quadro 8 totals, all ages (religiondots' parse of the same reports, sources/st.py there).
ALL_AGES = {"ST11": 7_324, "ST21": 69_454, "ST22": 17_161, "ST23": 6_031, "ST24": 14_652,
            "ST25": 19_365, "ST26": 44_752}
NATIONAL_ALL_AGES = 178_739
NATIONAL_1PLUS = 173_015

# The table is printed in two blocks of four languages; each block's header is anchored on
# its first language and the block's first row is the district (or national) total.
BLOCK1 = ["Português", "Fôrro", "Angolar", "Lunguié"]
BLOCK2 = ["Cabo verdiano", "Francês", "Inglês", "Outra(s) língua(s)"]
LANGS = BLOCK1 + BLOCK2
DENOM = "População 1+"

NUM = re.compile(r"^\d{1,3}(?:\.\d{3})*$|^\d+$")


def src(fname):
    for d in (RAW, RD_RAW):
        p = d / fname
        if p.exists() and p.stat().st_size > 500_000:
            return p
    raise SystemExit(f"{fname} not found in {RAW} or {RD_RAW}; run with --fetch")


def fetch():
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    for fname, remote in [(f, r) for f, _, r in DISTRICTS.values()] + [NATIONAL]:
        dest = RAW / fname
        if dest.exists() and dest.stat().st_size > 500_000:
            print(f"  have {fname}")
            continue
        url = BASE + urllib.parse.quote(DIR_2012 + remote)
        print("GET", url)
        r = requests.get(url, headers=UA, timeout=900)
        r.raise_for_status()
        body = r.content
        if not body.startswith(b"%PDF") or b"%%EOF" not in body[-4096:]:
            raise SystemExit(f"{fname}: {len(body):,} bytes with no %%EOF trailer")
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(body)
        os.replace(tmp, dest)


def _numbers(lines, start, want):
    vals, j = [], start
    while j < len(lines) and len(vals) < want:
        for tok in lines[j].split():
            if NUM.match(tok):
                vals.append(int(tok.replace(".", "")))
        j += 1
    if len(vals) != want:
        raise SystemExit(f"read {len(vals)} figures, wanted {want}: {vals}")
    return vals


def read_quadro10(path, label):
    """-> (population 1+, {language: Sim})."""
    import fitz
    doc = fitz.open(path)
    if doc.page_count == 0:
        raise SystemExit(f"{path}: zero pages")
    lines = []
    for i in range(doc.page_count):
        t = doc[i].get_text()
        if "Quadro 10" in t and "língua falada" in t:
            lines += [l.strip() for l in t.split("\n") if l.strip()]
    doc.close()
    if not lines:
        raise SystemExit(f"{label}: no Quadro 10 pages")
    out, totals = {}, set()
    for block, anchor in ((BLOCK1, "Lunguié"), (BLOCK2, "Inglês")):
        try:
            a = lines.index(anchor)
        except ValueError:
            raise SystemExit(f"{label}: Quadro 10 header {anchor!r} not found")
        vals = _numbers(lines, a + 1, 12)
        for k, lang in enumerate(block):
            tot, sim, nao = vals[3 * k: 3 * k + 3]
            if sim + nao != tot:
                raise SystemExit(f"{label} {lang}: Sim {sim} + Não {nao} != Total {tot}")
            totals.add(tot)
            out[lang] = sim
    if len(totals) != 1:
        raise SystemExit(f"{label}: Quadro 10's Total differs across languages: {totals}")
    return totals.pop(), out


def main():
    rows, dist = [], {}
    for pcode, (fname, name, _) in DISTRICTS.items():
        pop, sims = read_quadro10(src(fname), name)
        dist[pcode] = (name, pop, sims)
        under1 = 1 - pop / ALL_AGES[pcode]
        print(f"  {pcode} {name:<28} 1+ {pop:>7,}  (all ages {ALL_AGES[pcode]:,}, "
              f"under-one {under1:.1%})  " +
              "  ".join(f"{l[:4]} {sims[l]:,}" for l in LANGS))
        if not 0.025 <= under1 <= 0.045:
            raise SystemExit(f"{name}: 1+ population is {under1:.1%} below all ages")
        if sum(sims.values()) < pop - (pop - sims["Português"]):
            raise SystemExit(f"{name}: fewer mentions than people")

    npop, nsims = read_quadro10(src(NATIONAL[0]), "national")
    if npop != NATIONAL_1PLUS:
        raise SystemExit(f"national Quadro 10 total {npop:,}, expected {NATIONAL_1PLUS:,}")
    if sum(p for _, p, _ in dist.values()) != npop:
        raise SystemExit("the districts' 1+ populations do not sum to the national one")
    for l in LANGS:
        s = sum(d[2][l] for d in dist.values())
        if s != nsims[l]:
            raise SystemExit(f"{l}: districts sum to {s:,}, national report says {nsims[l]:,}")
    print(f"  the seven districts sum to the national Quadro 10 on all {len(LANGS)} languages "
          f"and on the 1+ total ({npop:,})")
    print("  national: " + ", ".join(f"{l} {nsims[l]:,} ({nsims[l] / npop:.1%})" for l in LANGS))

    for pcode, (name, pop, sims) in dist.items():
        rows.append(dict(geo_id=pcode, geo_level="district", geo_name=name,
                         source_category=DENOM, count=pop))
        for l in LANGS:
            rows.append(dict(geo_id=pcode, geo_level="district", geo_name=name,
                             source_category=l, count=sims[l]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["geo_id", "geo_level", "geo_name",
                                           "source_category", "count"])
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, OUT)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    else:
        main()
