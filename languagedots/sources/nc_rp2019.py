"""New Caledonia, Recensement de la population 2019 (INSEE-ISEE): speakers of each Kanak
language aged 15 and over, per commune.

    python sources/nc_rp2019.py [--fetch]

Writes data/normalized/nc.csv: one row per (commune, source_category), `count` as published.
geo_id is the commune's INSEE code (98801-98833), the `unit` of religiondots' NC layer.

THE QUESTION. Everyone aged 15 and over was asked whether they speak a Kanak language, only
understand one, or neither, and which language(s) they speak. Nothing is asked about French or
any other language (Wallisian, Futunian, Tahitian, Vietnamese, Javanese, Bislama...). ISEE files
29 answers as Kanak languages, among them Tayo (a French-based creole of Saint-Louis, Mont-Dore)
and Faga uvea (West Uvean, a Polynesian outlier on Ouvea).

THE TABLES (ISEE, "Organisation coutumiere kanak" page, https://www.isee.nc/organisation-coutumiere-kanak):
  langues-vernaculaires-locuteurs.xls
    'par commune-langue'   MENTIONS: speakers of each named language by commune of residence,
                           2019. "un locuteur peut parler une ou plusieurs langues"; people who
                           said they speak a Kanak language without naming it are not in it.
                           Its 'Total Locuteurs' row is per commune (see the categories below).
    'par commune'          PERSONS: distinct speakers of a Kanak language aged 15+, 1996-2019.
    'langue vernaculaire'  the national mentions per language, 1996-2019.
  rp2019-pop-logement-menages-communes.xls
    'P21'                  population 15+: speaks / only understands / no knowledge, by commune.
    'P01'                  population of all ages by commune.
  langues-vernaculaires-connaissance.xls
    'commune_2019'         population 15+: no knowledge / speaks or understands, by commune.

CATEGORIES written (source_category):
  <language label>         mentions, as 'par commune-langue' prints the label
  Total locuteurs          the 'Total Locuteurs' row: persons who named at least one language
                           (check 5 shows it is below the mentions and at most the speakers)
  Parle                    persons 15+ who speak a Kanak language (P21; = 'par commune' 2019)
  Comprend                 persons 15+ who only understand one (P21)
  Aucune connaissance      persons 15+ with neither (P21)
  Population 15+           P21's total
  Population               P01's total, all ages

CHECKS (the script stops unless all hold):
  1. 33 communes in every table, each name mapped to one INSEE code.
  2. 'par commune-langue': every language's communes sum to its TOTAL column, and the TOTAL
     column equals the 2019 column of 'langue vernaculaire'.
  3. 'par commune' 2019 equals P21's Parle in every commune, and both sum to 75,853, the
     distinct speakers the table's note gives.
  4. P21's Parle + Comprend equals 'connaissance' Parle ou comprend, and the two tables' 15+
     totals agree, in every commune.
  5. Per commune: Total locuteurs <= Parle, and Total locuteurs <= the sum of mentions, and
     every language's mentions <= Total locuteurs.
  6. P01's population equals the `pop` religiondots' NC layer carries for that INSEE code (the
     2019 census's own count), which is the name-to-code join checked against an independent
     source; and Population 15+ <= Population.
"""
import os
import sys
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "nc"
NORM = HERE / "data" / "normalized"
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
BASE = "https://www.isee.nc/sites/default/files/"
FILES = {
    "langues-vernaculaires-locuteurs.xls": "2025-09/langues-vernaculaires-locuteurs.xls",
    "langues-vernaculaires-connaissance.xls": "2025-09/langues-vernaculaires-connaissance.xls",
    "rp2019-pop-logement-menages-communes.xls": "2025-10/rp2019-pop-logement-menages-communes.xls",
}
DISTINCT = 75853          # "Au total, le nombre de locuteurs distincts est de 75853"

# INSEE commune codes (Kouaoua, created 1995, is 98833 out of alphabetical order). Folded names;
# the tables spell some communes differently (Bouloupari/Boulouparis, K-Gomen, Mt-Dore).
CODES = {
    "belep": "98801", "bouloupari": "98802", "boulouparis": "98802", "bourail": "98803",
    "canala": "98804", "dumbea": "98805", "farino": "98806", "hienghene": "98807",
    "houailou": "98808", "ile des pins": "98809", "ile des pins (l')": "98809",
    "kaala-gomen": "98810", "k-gomen": "98810", "kone": "98811", "koumac": "98812",
    "la foa": "98813", "lifou": "98814", "mare": "98815", "moindou": "98816",
    "mont-dore (le)": "98817", "mt-dore": "98817", "mont-dore": "98817", "noumea": "98818",
    "ouegoa": "98819", "ouvea": "98820", "paita": "98821", "poindimie": "98822",
    "ponerihouen": "98823", "pouebo": "98824", "pouembout": "98825", "poum": "98826",
    "poya": "98827", "sarramea": "98828", "thio": "98829", "touho": "98830", "voh": "98831",
    "yate": "98832", "kouaoua": "98833",
}
AGGREGATES = {"nouvelle-caledonie", "total", "province sud", "province nord",
              "province des iles", "province iles loyaute", "province des iles loyaute",
              "sud", "nord", "iles", "iles loyaute", "nord-ouest", "nord-est", "grand noumea",
              "sud rural"}


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, path in FILES.items():
        req = urllib.request.Request(BASE + path, headers=UA)
        data = urllib.request.urlopen(req, timeout=120).read()
        (RAW / name).write_bytes(data)
        print(f"  {name}: {len(data):,} bytes")


def fold(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace("\n", " ").split())


def code(name):
    f = fold(name)
    if f in CODES:
        return CODES[f]
    if f in AGGREGATES or f.startswith(("province", "total", "nouvelle")):
        return None
    raise SystemExit(f"nc: commune name not in CODES: {name!r}")


def names_of(head, communes):
    return {c: head[j] for j, c in communes.items()}


def num(v):
    return 0 if pd.isna(v) else int(round(float(v)))


def commune_rows(df, first_row, cols):
    """{code: [values]} for every row of `df` from `first_row` naming a commune in column 0."""
    out = {}
    for i in range(first_row, df.shape[0]):
        name = df.iat[i, 0]
        if not isinstance(name, str) or not name.strip():
            continue
        if all(pd.isna(df.iat[i, j]) for j in cols):
            continue                                   # a note under the table
        if fold(name) == "nouvelle-caledonie":
            break                                      # P21 has a 2014 table below 2019's
        c = code(name)
        if c is None:
            continue
        if c in out:
            raise SystemExit(f"nc: commune {name!r} twice")
        out[c] = [num(df.iat[i, j]) for j in cols]
    if len(out) != 33:
        raise SystemExit(f"nc: {len(out)} communes, expected 33")
    return out


def cell(v):
    """A header cell folded; a year read as 2019.0 comes back as '2019'."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return fold(v) if isinstance(v, (str, int)) else None


def header_row(df, text, col=None):
    for i in range(df.shape[0]):
        cells = df.iloc[i] if col is None else [df.iat[i, col]]
        if any(cell(v) == fold(text) for v in cells):
            return i
    raise SystemExit(f"nc: no row with {text!r}")


def main():
    if "--fetch" in sys.argv:
        fetch()
    loc = pd.ExcelFile(RAW / "langues-vernaculaires-locuteurs.xls")

    # ---- mentions by commune and language ----
    m = loc.parse("par commune-langue", header=None)
    h = header_row(m, "TOTAL")
    head = [str(v) for v in m.iloc[h]]
    if fold(head[-1]) != "total" or cell(m.iat[h, 0]) != "2019":
        raise SystemExit(f"nc: unexpected header {head}")
    communes = {j: code(head[j]) for j in range(1, len(head) - 1)}
    if sorted(communes.values()) != sorted(set(CODES.values())):
        raise SystemExit("nc: 'par commune-langue' columns are not the 33 communes")
    mentions, total_loc = {}, None
    for i in range(h + 1, m.shape[0]):
        label = m.iat[i, 0]
        if not isinstance(label, str) or label.startswith(("*", "NB", "Au total")):
            continue
        label = label.strip()
        vals = [num(m.iat[i, j]) for j in range(1, len(head))]
        if all(pd.isna(m.iat[i, j]) for j in range(1, len(head))):
            continue                                   # a "Groupe de ..." heading
        if fold(label) == "total locuteurs":
            total_loc = dict(zip(communes.values(), vals[:-1]))
            continue
        if sum(vals[:-1]) != vals[-1]:
            raise SystemExit(f"nc: {label}: communes sum {sum(vals[:-1])} != TOTAL {vals[-1]}")
        mentions[label] = (dict(zip(communes.values(), vals[:-1])), vals[-1])
    if total_loc is None or len(mentions) != 29:
        raise SystemExit(f"nc: {len(mentions)} languages (expected 29) or no Total Locuteurs row")

    # check 2: national per-language series
    lv = loc.parse("langue vernaculaire", header=None)
    hy = header_row(lv, "2019")
    ycol = [j for j in range(lv.shape[1]) if cell(lv.iat[hy, j]) == "2019"][0]
    nat = {}
    for i in range(hy + 1, lv.shape[0]):
        label = lv.iat[i, 0]
        if isinstance(label, str) and not pd.isna(lv.iat[i, ycol]) and label.strip() in mentions:
            nat[label.strip()] = num(lv.iat[i, ycol])
    if nat.keys() != mentions.keys():
        raise SystemExit(f"nc: national table labels differ: {set(mentions) ^ set(nat)}")
    bad = {k: (v[1], nat[k]) for k, v in mentions.items() if v[1] != nat[k]}
    if bad:
        raise SystemExit(f"nc: commune table TOTAL != national 2019 column: {bad}")
    print(f"  2. 29 languages; every row sums to its TOTAL and to the national 2019 column "
          f"({sum(nat.values()):,} mentions)")

    # check 3: distinct speakers by commune vs P21
    pc = loc.parse("par commune", header=None)
    hy = header_row(pc, "2019")
    ycol = [j for j in range(pc.shape[1]) if cell(pc.iat[hy, j]) == "2019"][0]
    speakers = commune_rows(pc, hy + 1, [ycol])
    rp = pd.ExcelFile(RAW / "rp2019-pop-logement-menages-communes.xls")
    p21 = rp.parse("P21", header=None)
    # Ensemble block: the last four columns, Parle / Comprend / Aucune / Total
    hdr = [fold(v) for v in p21.iloc[header_row(p21, "Total")]]
    if hdr[-4:] != ["parle une langue kanak", "comprend une langue kanak",
                    "aucune connaissance", "total"]:
        raise SystemExit(f"nc: P21 header {hdr[-4:]}")
    n = p21.shape[1]
    p21v = commune_rows(p21, 3, [n - 4, n - 3, n - 2, n - 1])
    for c, v in p21v.items():
        if v[0] + v[1] + v[2] != v[3]:
            raise SystemExit(f"nc: P21 {c}: parts do not sum to total")
        if v[0] != speakers[c][0]:
            raise SystemExit(f"nc: {c}: P21 Parle {v[0]} != 'par commune' {speakers[c][0]}")
    tot = sum(v[0] for v in p21v.values())
    if tot != DISTINCT:
        raise SystemExit(f"nc: speakers sum {tot} != {DISTINCT}")
    print(f"  3. P21 Parle = 'par commune' 2019 in all 33 communes; sum {tot:,} = the note's "
          f"{DISTINCT:,}")

    # check 4: connaissance table
    kn = pd.ExcelFile(RAW / "langues-vernaculaires-connaissance.xls").parse("commune_2019",
                                                                           header=None)
    n = kn.shape[1]
    knv = commune_rows(kn, 7, [n - 3, n - 2, n - 1])
    for c, v in knv.items():
        if v[1] != p21v[c][0] + p21v[c][1] or v[2] != p21v[c][3]:
            raise SystemExit(f"nc: {c}: connaissance {v} vs P21 {p21v[c]}")
    print("  4. 'connaissance' Parle ou comprend = P21 Parle + Comprend, and the 15+ totals "
          "agree, in all 33 communes")

    # check 5. 'Total Locuteurs' turns out to be the sum of the mentions, not a count of persons
    # (65,185 nationally, the same as the 29 languages' totals), so how many speakers named
    # two languages is not published. Per commune, mentions above speakers would show it.
    over = []
    for c in total_loc:
        s = sum(v[0][c] for v in mentions.values())
        if total_loc[c] != s:
            raise SystemExit(f"nc: {c}: Total Locuteurs {total_loc[c]} != sum of mentions {s}")
        if s > p21v[c][0]:
            over.append(f"{names_of(head, communes)[c]} {s}>{p21v[c][0]}")
    s_all = sum(v[1] for v in mentions.values())
    print(f"  5. 'Total Locuteurs' = the sum of mentions in all 33 communes ({s_all:,}), below the "
          f"{DISTINCT:,} speakers: at least {DISTINCT - s_all:,} named no language. Communes "
          f"with more mentions than speakers: {over or 'none'}")

    # check 6: population, and the name-to-code join against religiondots' layer
    p01 = rp.parse("P01", header=None)
    pop = {c: v[0] for c, v in commune_rows(p01, 4, [p01.shape[1] - 1]).items()}
    import geopandas as gpd
    rd = gpd.read_file(RD_GEO / "nc" / "nc_units.gpkg", ignore_geometry=True)
    rdpop = dict(zip(rd["unit"].astype(str), rd["pop"].astype(int)))
    if rdpop != pop:
        diff = {c: (pop.get(c), rdpop.get(c)) for c in set(pop) | set(rdpop)
                if pop.get(c) != rdpop.get(c)}
        raise SystemExit(f"nc: P01 population != religiondots' nc_units pop: {diff}")
    for c in pop:
        if p21v[c][3] > pop[c]:
            raise SystemExit(f"nc: {c}: 15+ exceeds population")
    print(f"  6. P01 population = religiondots' nc_units pop for all 33 codes; "
          f"{sum(pop.values()):,} people, {sum(v[3] for v in p21v.values()):,} aged 15+")

    names = names_of(head, communes)
    rows = []
    for c in sorted(pop):
        base = dict(geo_id=c, geo_name=names[c], geo_level="commune")
        for label, (vals, _) in mentions.items():
            rows.append({**base, "source_category": label, "count": vals[c]})
        rows.append({**base, "source_category": "Total locuteurs", "count": total_loc[c]})
        for k, lab in enumerate(["Parle", "Comprend", "Aucune connaissance", "Population 15+"]):
            rows.append({**base, "source_category": lab, "count": p21v[c][k]})
        rows.append({**base, "source_category": "Population", "count": pop[c]})
    out = pd.DataFrame(rows)
    NORM.mkdir(parents=True, exist_ok=True)
    out.to_csv(NORM / "nc.csv", index=False, encoding="utf-8")
    print(f"  wrote {NORM / 'nc.csv'}: {len(out):,} rows, {out['geo_id'].nunique()} communes")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    print("  1. commune names map to 33 INSEE codes in every table (checked as each is read)")
    main()
