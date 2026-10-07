"""Suriname, Census 7 (2004): the language most spoken in the household, per ressort
-> data/normalized/sr.csv.

    python sources/sr_census.py [--fetch]

THE TABLE. ABS (Algemeen Bureau voor de Statistiek), `census-profile-on-ressort-level.xls`,
sheet "POPULATION BY RESSORT", block 11 "Most Spoken Language in the household": the number of
HOUSEHOLDS by the language their members usually speak to one another (Census 7 Volume 4's
definition: "De taal die doorgaans gesproken wordt door de verschillende leden van het
huishouden in onderlinge communicatie"), for 62 ressorten and the nation. Fifteen rows: Dutch,
Sranan tongo, Sarnami, Javanese, Arowaks, Caraib (the last two under a printed heading
"Indigenous languages"), Saramaccaans, Aucaans, Paramaccaans ("Marron languages"), Chinese,
Portugese, English, French, Other, Unknown. The same sheet carries each ressort's population
(block 3) and number of households (block 10). Religiondots draws block 5 of this file.

WHY 2004 AND NOT 2012. Census 8 (2012) asked the same household question, but published it
nationally in six rows (Volume 3, table HWG-06a: Dutch, Sranan, Sarnami, Javanese, the Maroon
languages pooled, everything else pooled). By district it has full tables for Paramaribo and
Wanica only (Districtsresultaten Vol. I, 11 rows); Vol. II shows Nickerie, Coronie, Saramacca,
Commewijne and Para as a chart whose text layer holds only each district's largest language;
Vol. III (census8dis3.pdf: Marowijne, Brokopondo, Sipaliwini) has no language table.
2004 is the only whole-country sub-national language table, at the finest grain ABS publishes.
Census 9 (2024-25) has published no results yet.

HOUSEHOLDS INTO PEOPLE. The table counts households, and a household is drawn as its members,
all in its language (as Paraguay's and Namibia's household questions are drawn). Ressort
tables give no household sizes, so each household is weighted by the national mean size of
households of its language, from Census 7 Volume 4, Tabel 02 ("Aantal huishoudens naar
huishoudomvang en naar meest gesproken taal in het huishouden"), which pools the indigenous and
the Maroon languages, so those two pools carry one mean each. Its open "10 & meer" column is
given the one mean size that makes the table's people sum to Census 7's non-institutional
population, 486,907 (Volume 1; 123,463 households, mean 3.94, Volume 4). Then each ressort's
people are scaled to its census population, so every ressort keeps its own total:

    people[r, L] = households[r, L] * size[L] * population[r] / sum_L(households[r, L] * size[L])

The scaling includes Unknown, whose people are then not drawn. Every row is `derived`. Means
run from 3.1 (French) and 3.6 (Sranan, Javanese) to 4.3 (Maroon) and 5.2 (Portuguese), so this
moves 10,154 people (2.1%) between languages compared with plain household shares.

CHECKS, all asserted:
  1. every ressort's fifteen languages sum to its own printed number of households (block 10),
     and the national column to 123,463;
  2. the 62 ressorten sum to the national column, row by row;
  3. A SECOND PUBLICATION: Census 7 Volume 4, Tabel 01 and Tabel 02 (pdf page 32), print the
     national households per language (indigenous and Maroon pooled); their row totals must
     equal the xls's national column;
  4. Tabel 02's size columns sum to its printed totals;
  5. the 62 ressort populations sum to 492,829.

GEOGRAPHY. The 62 ressorten in the xls's column order, with their district and COD-AB ADM2
p-code, are religiondots' (sources/sr.py `RESSORTEN`, read-only; its sources/sr_geo.py derives
the pairing from names and asserts it), copied here and asserted against the header names.
"""
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "sr"
OUT = ROOT / "data" / "normalized" / "sr.csv"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
BASE = "https://statistics-suriname.org/wp-content/uploads/"
FILES = {
    "census-profile-on-ressort-level.xls": "2019/03/census-profile-on-ressort-level.xls",
    "SURINAME-CENSUS-2004-VOLUME-4-HOUSEHOLDS-FAMILIES-AND-HOUSING.pdf":
        "2019/10/SURINAME-CENSUS-2004-VOLUME-4-HOUSEHOLDS-FAMILIES-AND-HOUSING.pdf",
    # read for the record, not parsed: the 2012 tables that were not used (see WHY 2004)
    "Publicatie-Census-8-Volume-3-Huishoudens-Gezinnen-en-Woonverblijven-Milieu-Criminaliteit.pdf":
        "2019/05/Publicatie-Census-8-Volume-3-Huishoudens-Gezinnen-en-Woonverblijven-Milieu-"
        "Criminaliteit.pdf",
    "presentatie-districts-resultaten-vol1-070314.pdf":
        "2019/03/presentatie-districts-resultaten-vol1-070314.pdf",
    "districtsresultaten_volii_finale.pdf": "2019/03/districtsresultaten_volii_finale.pdf",
    "census8dis3.pdf": "2019/03/census8dis3.pdf",
}
SHEET = "POPULATION BY RESSORT"
NAT_POP = 492_829
NAT_HH = 123_463
NONINST_POP = 486_907       # Census 7 Volume 1, p.16: the non-institutional population

# Block 11's row labels as printed (col 1, whitespace-collapsed; two carry their group heading)
LANGS = ["Dutch", "Sranan tongo", "Sarnami", "Javanese", "Arowaks Indigenous languages",
         "Caraib", "Saramaccaans", "Aucaans Marron languages", "Paramaccaans", "Chinese",
         "Portugese", "English", "French", "Other", "Unknown"]
# ... written to the CSV without the bled-in headings
CLEAN = {"Arowaks Indigenous languages": "Arowaks", "Aucaans Marron languages": "Aucaans"}
# Volume 4's rows, and which of block 11's rows each pools
V4 = {"Nederlands": ["Dutch"], "Sranan tongo": ["Sranan tongo"], "Sarnami": ["Sarnami"],
      "Javaans": ["Javanese"], "Inheemse taal": ["Arowaks", "Caraib"],
      "Marrontaal": ["Saramaccaans", "Aucaans", "Paramaccaans"], "Chinees": ["Chinese"],
      "Portugees": ["Portugese"], "Engels": ["English"], "Frans": ["French"],
      "Andere": ["Other"], "Onbekend": ["Unknown"]}

# (district, ressort as ABS prints it, COD-AB ADM2_PCODE), in the xls's column order.
# Copied from ../religiondots/sources/sr.py RESSORTEN (read-only).
RESSORTEN = [
    ("Paramaribo", "Blauwgrond", "SR0702"), ("Paramaribo", "Rainville", "SR0709"),
    ("Paramaribo", "Munder", "SR0707"), ("Paramaribo", "Centrum", "SR0703"),
    ("Paramaribo", "Beekhuizen", "SR0701"), ("Paramaribo", "Weg naar Zee", "SR0711"),
    ("Paramaribo", "Welgelegen (Par'bo)", "SR0712"), ("Paramaribo", "Tammenga", "SR0710"),
    ("Paramaribo", "Flora", "SR0704"), ("Paramaribo", "Latour", "SR0705"),
    ("Paramaribo", "Pontbuiten", "SR0708"), ("Paramaribo", "Livorno", "SR0706"),
    ("Wanica", "Kwatta", "SR1005"), ("Wanica", "Saramacca Polder", "SR1007"),
    ("Wanica", "Koewarasan", "SR1004"), ("Wanica", "De Nieuwe Grond", "SR1001"),
    ("Wanica", "Lelydorp", "SR1006"), ("Wanica", "Houttuin", "SR1003"),
    ("Wanica", "Domburg", "SR1002"),
    ("Nickerie", "Wageningen", "SR0504"), ("Nickerie", "Groot Henar", "SR0501"),
    ("Nickerie", "Oostelijke Polders", "SR0503"), ("Nickerie", "Nieuw Nickerie", "SR0502"),
    ("Nickerie", "Westelijke Polders", "SR0505"),
    ("Coronie", "Welgelegen (Coronie)", "SR0303"), ("Coronie", "Totness", "SR0302"),
    ("Coronie", "Johanna Maria", "SR0301"),
    ("Saramacca", "Calcutta", "SR0801"), ("Saramacca", "Tijgerkreek", "SR0805"),
    ("Saramacca", "Groningen", "SR0802"), ("Saramacca", "Kampong Baroe", "SR0804"),
    ("Saramacca", "Wayambo", "SR0806"), ("Saramacca", "Jarikaba", "SR0803"),
    ("Commewijne", "Margaretha", "SR0203"), ("Commewijne", "Bakkie", "SR0202"),
    ("Commewijne", "Nieuw Amsterdam", "SR0205"), ("Commewijne", "Alkmaar", "SR0201"),
    ("Commewijne", "Tamanredjo", "SR0206"), ("Commewijne", "Meerzorg", "SR0204"),
    ("Marowijne", "Moengo", "SR0403"), ("Marowijne", "Wanhatti", "SR0406"),
    ("Marowijne", "Galibi", "SR0402"), ("Marowijne", "Moengo Tapoe", "SR0404"),
    ("Marowijne", "Albina", "SR0401"), ("Marowijne", "Patamacca", "SR0405"),
    ("Para", "Para Noord", "SR0603"), ("Para", "Para Oost", "SR0604"),
    ("Para", "Para Zuid", "SR0605"), ("Para", "Bigi Poika", "SR0601"),
    ("Para", "Carolina", "SR0602"),
    ("Brokopondo", "Kwakoegron", "SR0104"), ("Brokopondo", "Marchallkreeek", "SR0105"),
    ("Brokopondo", "Klaaskreek", "SR0103"), ("Brokopondo", "Brokopondo Centrum", "SR0102"),
    ("Brokopondo", "Brownsweg", "SR0101"), ("Brokopondo", "Sarakreek", "SR0106"),
    ("Sipaliwini", "Tapanahony", "SR0906"), ("Sipaliwini", "Boven-Suriname", "SR0903"),
    ("Sipaliwini", "Boven-Saramacca", "SR0902"), ("Sipaliwini", "Boven-Coppename", "SR0901"),
    ("Sipaliwini", "Kabalebo", "SR0905"), ("Sipaliwini", "Coeroeni", "SR0904"),
]


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, rel in FILES.items():
        dest = RAW / name
        if dest.exists() and dest.stat().st_size > 10_000:
            print("have", name)
            continue
        req = urllib.request.Request(BASE + rel, headers={"User-Agent": UA})
        data = urllib.request.urlopen(req, timeout=300).read()
        magic = b"\xd0\xcf\x11\xe0" if name.endswith(".xls") else b"%PDF-"
        if not data.startswith(magic):
            raise SystemExit(f"{name}: not a {'.xls' if name.endswith('.xls') else 'PDF'}, "
                             f"starts {data[:8]!r}")
        dest.write_bytes(data)
        print(f"GET {name} {len(data):,} B")


def _norm(s):
    return " ".join(str(s or "").split())


def read_xls():
    import xlrd
    sh = xlrd.open_workbook(RAW / "census-profile-on-ressort-level.xls").sheet_by_name(SHEET)
    names = [_norm(sh.cell_value(3, c)) for c in range(3, sh.ncols)]
    want = [r for _, r, _ in RESSORTEN]
    if names != want:
        bad = [(i, a, b) for i, (a, b) in enumerate(zip(names, want)) if a != b]
        raise SystemExit(f"ressort columns differ from RESSORTEN: {bad[:3]} (len {len(names)})")
    if _norm(sh.cell_value(3, 2)) != "Total":
        raise SystemExit("column 2 is not the national Total")

    def find(label, after=0):
        for r in range(after, sh.nrows):
            if _norm(sh.cell_value(r, 1)) == label:
                return r
        raise SystemExit(f"no row {label!r}")

    r_lang = find("Most Spoken Language in the household")
    r_pop = find("Total", find("Population:"))
    r_hh = find("Number of Households")
    cols = range(2, 3 + len(RESSORTEN))
    ids = ["SR"] + [p for _, _, p in RESSORTEN]
    rows = []
    pop = {i: int(sh.cell_value(r_pop, c)) for i, c in zip(ids, cols)}
    hh = {i: int(sh.cell_value(r_hh, c)) for i, c in zip(ids, cols)}
    for k, lab in enumerate(LANGS):
        r = r_lang + 1 + k
        got = _norm(sh.cell_value(r, 1))
        if got != lab:
            raise SystemExit(f"row {r}: expected {lab!r}, read {got!r}")
        for i, c in zip(ids, cols):
            v = sh.cell_value(r, c)
            if not isinstance(v, float) or v != int(v) or v < 0:
                raise SystemExit(f"{i}/{lab}: {v!r} is not a count")
            rows.append((i, CLEAN.get(lab, lab), int(v)))
    df = pd.DataFrame(rows, columns=["geo_id", "source_category", "households"])
    return df, pop, hh


def read_v4():
    """Volume 4, pdf page 32: Tabel 01's row totals and Tabel 02's sizes by language."""
    import fitz
    doc = fitz.open(RAW / "SURINAME-CENSUS-2004-VOLUME-4-HOUSEHOLDS-FAMILIES-AND-HOUSING.pdf")
    text = doc[31].get_text()
    i01, i02 = text.index("Tabel 01."), text.index("Tabel 02.")
    t01, t02 = text[i01:i02], text[i02:]
    labels = list(V4) + ["Totaal"]

    def rows(block, ncols, lab_alias):
        out = {}
        toks = [t.strip() for t in block.splitlines() if t.strip()]
        for lab in labels:
            want = lab_alias.get(lab, lab)
            # the row label as a line of its own, the LAST occurrence (headers repeat names)
            pos = [k for k, t in enumerate(toks) if t == want]
            if not pos:
                raise SystemExit(f"Volume 4: no row {want!r}")
            k = pos[-1]
            nums = toks[k + 1:k + 1 + ncols]
            if not all(re.fullmatch(r"\d+", n) for n in nums):
                raise SystemExit(f"Volume 4 row {want!r}: {nums}")
            out[lab] = [int(n) for n in nums]
        return out

    t1 = rows(t01, 14, {"Marrontaal": "Marron taal"})      # Tabel 01 spells it in two words
    t2 = rows(t02, 11, {})
    return t1, t2


def main():
    if "--fetch" in sys.argv:
        fetch()
    df, pop, hh = read_xls()
    w = df.pivot(index="geo_id", columns="source_category", values="households")
    langs = [CLEAN.get(lab, lab) for lab in LANGS]
    w = w[langs]
    ids = [p for _, _, p in RESSORTEN]

    # 1. languages sum to each unit's households
    s = w.sum(axis=1)
    bad = {i: (int(s[i]), hh[i]) for i in w.index if s[i] != hh[i]}
    assert not bad, f"language rows != households: {bad}"
    assert hh["SR"] == NAT_HH, hh["SR"]
    # 2. ressorten sum to the nation, row by row
    diff = w.loc[ids].sum() - w.loc["SR"]
    assert (diff == 0).all(), diff[diff != 0]
    # 5. populations
    assert sum(pop[i] for i in ids) == pop["SR"] == NAT_POP, pop["SR"]
    print(f"check 1-2, 5: 62 ressorten, {NAT_HH:,} households, {NAT_POP:,} people; exact")

    # 3-4. Volume 4
    t1, t2 = read_v4()
    nat = w.loc["SR"]
    for v4, mine in V4.items():
        want = int(sum(nat[m] for m in mine))
        assert t1[v4][-1] == want, (v4, t1[v4][-1], want)
        assert t2[v4][-1] == want, (v4, t2[v4][-1], want)
        assert sum(t2[v4][:-1]) == want, (v4, "Tabel 02 row sum")
    assert t1["Totaal"][-1] == t2["Totaal"][-1] == NAT_HH
    for k in range(10):
        assert sum(t2[v][k] for v in V4) == t2["Totaal"][k], ("Tabel 02 column", k)
    print("check 3-4: Volume 4 Tabel 01 and 02 national rows equal the xls; Tabel 02 adds up")

    # household sizes by language; the open 10+ bin solved from the non-institutional total
    tot = t2["Totaal"]
    closed = sum((k + 1) * tot[k] for k in range(9))
    ten = (NONINST_POP - closed) / tot[9]
    assert 10 <= ten <= 14, ten
    size = {}
    for v4, mine in V4.items():
        r = t2[v4]
        people = sum((k + 1) * r[k] for k in range(9)) + ten * r[9]
        for m in mine:
            size[m] = people / r[10]
    print(f"open 10+ bin: {ten:.2f} people per household; mean size by language: "
          + ", ".join(f"{k} {v:.2f}" for k, v in size.items()))

    # people
    est = w.mul(pd.Series(size))
    scale = pd.Series(pop).reindex(est.index) / est.sum(axis=1)
    people = est.mul(scale, axis=0)
    plain = w.mul(pd.Series(pop).reindex(w.index) / w.sum(axis=1), axis=0)
    moved = (people.loc[ids] - plain.loc[ids]).abs().sum().sum() / 2
    print(f"household-size weighting moves {moved:,.0f} people ({moved / NAT_POP:.1%}) "
          "against plain household shares")

    meta = {p: (d, r) for d, r, p in RESSORTEN}
    out = []
    for gid in ["SR"] + ids:
        lvl, dist, name = (("national", "", "Suriname") if gid == "SR"
                           else ("ressort", *meta[gid]))
        for lab in langs:
            out.append(dict(geo_id=gid, geo_level=lvl, district=dist, geo_name=name,
                            source_category=lab, households=int(w.at[gid, lab]),
                            count=round(float(people.at[gid, lab]), 2)))
    out = pd.DataFrame(out)
    # national row = sum of ressorten (scaled per ressort, so recompute rather than scale)
    natp = out[out.geo_level == "ressort"].groupby("source_category")["count"].sum()
    out.loc[out.geo_level == "national", "count"] = (
        out.loc[out.geo_level == "national", "source_category"].map(natp).round(2))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} rows)")
    print((natp / NAT_POP * 100).round(2).sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
