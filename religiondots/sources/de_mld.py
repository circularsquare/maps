"""Germany: branch shares of each Bundesland's Muslims, from BAMF's *Muslimisches Leben in
Deutschland 2020* (MLD 2020).

Writes data/normalized/de_mld.csv: per Land (the two-digit ARS prefix de_ess.csv uses), the share
of its Muslims that are Sunni, Shi'a, Alevi and Ahmadi. countries/de.py::_de_split_islam applies
them to the ESS-modelled `islam` rows, which stay `modelled`. Whatever MLD leaves unnamed (other,
just Muslim, don't know) stays on `islam` (spec §2.7a).

Approved by Anita 2026-10-16 (research: unspecified_splits/islam.md §3).

THE SOURCE
----------
Stichs, Pfündel et al., *Muslimisches Leben in Deutschland 2020*, BAMF Forschungsbericht 38
(2021), data/raw/de/mld2020/fb38-muslimisches-leben.pdf, from
https://www.bamf.de/SharedDocs/Anlagen/DE/Forschung/Forschungsberichte/fb38-muslimisches-leben.pdf
Telephone survey 2020, 5,120 respondents 16+ (4,538 with a migration background from 23
Muslim-majority countries), 11,243 household members; weighted; projected onto Mikrozensus 2019.

  Table 23 (p. 38)  Muslims with a migration background in 2019, middle estimate, thousands, by
                    origin region: Mittlerer Osten (Afghanistan, Bangladesh, Iran, Pakistan) 481;
                    Naher Osten 1,050; Nordafrika 417; Südosteuropa 1,046; Türkei 2,462.
  Table 26 (p. 53)  each region's Muslims over the Länder, % (Bremen with Hamburg, Brandenburg
                    with Mecklenburg-Vorpommern, as printed).
  Table 32 (p. 59)  Glaubensrichtung by origin region, % of Muslims (question v201, n 3,398).

A Land's branch shares are the Muslim-weighted mean of Table 32's columns, the weights being
Table 23 x Table 26. Every input is a printed BAMF number.

WHAT IT CANNOT SEE
------------------
MLD's universe is people with a migration background from those 23 countries: converts and
Muslims from elsewhere (Chechnya, West Africa) are outside it. Table 32 is respondents 16+,
Table 26 all household members. The eastern Länder rest on a few dozen respondents each.

ALEVIS go to `alevism`, the precedent the map already follows wherever a source names them: the
2021 census of England and Wales (write-in `Alevi`, taxonomy/uk2021.py) and the foreign half of
France, Belgium, Sweden, Norway, Denmark, Finland, Italy, Greece and Spain (Turkish nationals,
taxonomy/origin_religion.py). MLD names `Alevitisch` as its own answer, so it is filed there too.
Bulgaria's census Kızılbaş are on `islam.shia` because that census's cell says Shia; that is the
same rule (record the source's category, spec §2.7), not a different one.

Usage:
    python sources/de_mld.py             write data/normalized/de_mld.csv
    python sources/de_mld.py --dry-run   report, write nothing
"""

import os
import re
import shutil
import subprocess
import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(HERE, "data", "raw", "de", "mld2020", "fb38-muslimisches-leben.pdf")
OUT = os.path.join(HERE, "data", "normalized", "de_mld.csv")

REGIONS = ["Mittlerer Osten", "Naher Osten", "Nordafrika", "Südosteuropa", "Türkei"]

# Table 23, middle estimate, thousands.
T23 = {"Mittlerer Osten": 481, "Naher Osten": 1050, "Nordafrika": 417, "Südosteuropa": 1046,
       "Türkei": 2462}
T23_TOTAL = 5457

# Table 32, % of Muslims, columns in REGIONS order, then MLD's own all-regions column.
T32 = {
    "Ahmadi":         [10.9, 0.2, 0.2, 0.3, 0.4, 1.3],
    "Alevitisch":     [1.1, 0.6, 0.1, 1.9, 16.9, 9.5],
    "Schiitisch":     [27.3, 5.5, 0.5, 2.8, 1.2, 4.4],
    "Sunnitisch":     [55.7, 89.3, 90.6, 54.9, 71.9, 71.7],
    "Sonstige":       [0.2, 0.3, 1.1, 0.2, 0.9, 0.6],
    "Unspezifiziert": [0.6, 0.2, 0.1, 3.1, 0.7, 1.0],
    "Weiß nicht":     [4.2, 3.9, 7.5, 36.9, 8.1, 11.6],
}
# The four that name a branch, and the category string each is written as (taxonomy/de2022.py).
NAMED = {"Sunnitisch": "MLD 2020: Sunnitisch", "Schiitisch": "MLD 2020: Schiitisch",
         "Alevitisch": "MLD 2020: Alevitisch", "Ahmadi": "MLD 2020: Ahmadi"}

# Table 26, % of each region's Muslims living in the Land, columns in REGIONS order. Keys are the
# ARS Land codes each printed row covers.
T26 = {
    ("08",): ("Baden-Württemberg", [7.5, 11.1, 8.6, 27.9, 18.3]),
    ("09",): ("Bayern", [9.3, 10.0, 9.0, 19.0, 10.7]),
    ("11",): ("Berlin", [3.2, 5.1, 2.7, 2.6, 6.8]),
    ("02", "04"): ("Bremen und Hamburg", [12.9, 5.8, 1.7, 3.0, 4.4]),
    ("06",): ("Hessen", [25.1, 4.3, 23.4, 7.4, 10.7]),
    ("03",): ("Niedersachsen", [7.9, 10.9, 4.8, 5.1, 5.6]),
    ("05",): ("Nordrhein-Westfalen", [15.3, 27.9, 42.1, 26.7, 36.1]),
    ("07",): ("Rheinland-Pfalz", [5.5, 3.6, 5.7, 4.1, 4.2]),
    ("10",): ("Saarland", [0.8, 4.1, 0.1, 0.5, 1.0]),
    ("01",): ("Schleswig-Holstein", [4.4, 4.7, 0.9, 1.9, 1.6]),
    ("12", "13"): ("Brandenburg und Mecklenburg-Vorpommern", [1.3, 2.8, 0.1, 0.8, 0.2]),
    ("14",): ("Sachsen", [1.0, 1.6, 0.5, 0.3, 0.2]),
    ("15",): ("Sachsen-Anhalt", [1.8, 5.1, 0.4, 0.5, 0.0]),
    ("16",): ("Thüringen", [4.3, 2.9, 0.1, 0.5, 0.3]),
}
# Table 26's printed subtotals, West incl. Berlin and East, per region.
T26_WEST = [91.6, 87.6, 98.9, 98.0, 99.3]
T26_EAST = [8.4, 12.4, 1.1, 2.1, 0.7]
EAST = {"12", "13", "14", "15", "16"}


def _nums(text):
    return [float(x.replace(".", "").replace(",", ".")) if "," in x else float(x.replace(".", ""))
            for x in re.findall(r"\d{1,3}(?:\.\d{3})*(?:,\d)?", text)]


def check_source():
    """Re-read the three tables' numbers off the PDF with pdftotext, where it is installed."""
    if not os.path.exists(PDF):
        raise SystemExit(f"missing {PDF}; download it from the URL in this file's docstring")
    exe = shutil.which("pdftotext")
    if exe is None:
        print("  !! pdftotext not on PATH; the transcription is checked by its sums only")
        return
    text = subprocess.run([exe, "-layout", PDF, "-"], capture_output=True, check=True
                          ).stdout.decode("latin-1")
    # Table 32: every printed cell, as a multiset, between its caption and its source line.
    i = text.index("Tabelle 32: Glaubensrichtungen")
    block = text[i:text.index("Quelle", i)]
    got = sorted(_nums(block.split("(in Prozent)", 1)[1]))
    want = sorted([v for row in T32.values() for v in row] + [100.0] * 6)
    if got != want:
        raise SystemExit(f"Table 32 on the PDF does not match T32:\n  pdf  {got}\n  here {want}")
    # Table 26: the same, cells plus the printed subtotals and the `Insgesamt` column.
    i = text.index("Tabelle 26: Verteilung")
    block = text[i:text.index("Quelle", i)]
    cells = [v for _, row in T26.values() for v in row] + T26_WEST + T26_EAST + [100.0] * 5
    got = sorted(_nums(block.split("(in Prozent)", 1)[1]))
    missing = [v for v in cells if v not in got]
    if missing or len(got) - len(cells) != 17:      # the 17 printed `Insgesamt` cells besides
        raise SystemExit(f"Table 26 on the PDF does not match T26 (missing {missing}, "
                         f"{len(got)} numbers against {len(cells)} + 17)")
    # Table 23: each region's subtotal row prints `<MZ> <min> <mid> <max>`.
    i = text.index("Tabelle 23: Anzahl")
    block = text[i:text.index("Quelle", i)]
    for region, mid in T23.items():
        m = re.search(rf"^\s*{region}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", block,
                      re.M)
        if not m or float(m.group(3).replace(".", "")) != mid:
            raise SystemExit(f"Table 23: {region} middle estimate is not {mid} on the PDF")
    print("  OK Tables 23, 26 and 32 re-read off the PDF")


def build():
    check_source()
    for k, row in T32.items():
        if len(row) != 6:
            raise SystemExit(f"T32 {k} has {len(row)} cells")
    for j, region in enumerate(REGIONS + ["all"]):
        s = sum(row[j] for row in T32.values())
        if abs(s - 100) > 0.25:
            raise SystemExit(f"Table 32 {region} sums to {s:.1f}")
    for j, region in enumerate(REGIONS):
        s = sum(row[j] for _, row in T26.values())
        w = sum(row[j] for codes, (_, row) in T26.items() if not set(codes) & EAST)
        if abs(s - 100) > 0.5 or abs(w - T26_WEST[j]) > 0.35:
            raise SystemExit(f"Table 26 {region}: column {s:.1f}, West {w:.1f} against "
                             f"printed {T26_WEST[j]}")
    if abs(sum(T23.values()) - T23_TOTAL) > 2:
        raise SystemExit("Table 23 regions do not sum to its total")
    covered = sorted(c for codes in T26 for c in codes)
    if covered != [f"{i:02d}" for i in range(1, 17)]:
        raise SystemExit(f"Table 26 rows cover Länder {covered}, not all sixteen")

    rows, nat = [], {k: 0.0 for k in T32}
    print(f"\n  {'Land':<40}{'Muslims k':>10}" + "".join(f"{k[:10]:>12}" for k in NAMED))
    for codes, (name, dist) in T26.items():
        m = [T23[r] * dist[j] / 100 for j, r in enumerate(REGIONS)]
        mt = sum(m)
        shares = {k: sum(m[j] * T32[k][j] for j in range(5)) / mt / 100 for k in T32}
        for k in T32:
            nat[k] += shares[k] * mt
        print(f"  {name:<40}{mt:>10.0f}" + "".join(f"{100 * shares[k]:>11.1f}%" for k in NAMED))
        for code in codes:
            for k, cat in NAMED.items():
                rows.append(dict(geo_id=code, geo_level="bundesland", geo_name=name,
                                 source_category=cat, share=round(shares[k], 6),
                                 muslims_k_mld=round(mt, 1), year=2020,
                                 source_id="bamf_mld_2020",
                                 note=f"MLD 2020 Tables 23 x 26 x 32; Table 26 row "
                                      f"'{name}'"))
    tot = sum(T23.values())
    print("\n  national, these weights against MLD's own all-regions column:")
    for k in NAMED:
        print(f"      {k:<12} {100 * nat[k] / tot:5.1f}%   (MLD {T32[k][5]}%)")
        # MLD's column weights respondents, these weight projected people; Alevi differs most
        if abs(100 * nat[k] / tot - T32[k][5]) > 1.5:
            raise SystemExit(f"{k}: the weighted national share is more than 1.5 points from "
                             "MLD's own; the transcription is probably wrong")
    out = pd.DataFrame(rows)
    named = out.groupby("geo_id")["share"].sum()
    if (named >= 1).any():
        raise SystemExit("a Land's named shares reach 100%")
    return out


def main():
    out = build()
    if "--dry-run" in sys.argv:
        print("  --dry-run: nothing written")
        return
    tmp = OUT + ".part"
    out.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, OUT)
    print(f"  wrote {OUT} ({len(out)} rows, {out['geo_id'].nunique()} Länder)")


if __name__ == "__main__":
    main()
