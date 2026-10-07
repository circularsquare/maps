"""Côte d'Ivoire RGPH 2021, "langue la plus parlée" (the national language each person speaks
most), by région -> data/normalized/ci.csv.

    python sources/ci_rgph.py [--fetch]

SOURCE. ANStat, RGPH 2021 *Rapport thématique tome 1: État et structure de la population*
(151 pages), the volume religiondots draws Côte d'Ivoire's religion from. Live at
https://www.anstat.ci/assets/publications/files/rgpg_tom1.pdf, which is behind Cloudflare and
403 to every script; the Wayback Machine has it (religiondots/sources/ci.md §1-2: take the
LARGEST capture, the first-listed one is cut at exactly 2^20 bytes and opens anyway). --fetch
copies religiondots' verified download when it is there (read-only) and otherwise takes the
archive's copy itself.

QUESTION (§1.2.1.7 of the tome): "les langues locales parlées désignent les langues nationales
(par exemple Guéré, Baoulé, Bété et Malinké) utilisées par les Ivoiriens et les non-Ivoiriens
âgés de 2 ans et plus pour communiquer, distinctement du français qui constitue la langue
officielle." One answer each: the national (Ivorian) language the person speaks most. French is
not an answer; a person who speaks no Ivorian language is "Aucune langue nationale parlée".
Every table published is for people of IVORIAN NATIONALITY only (21,295,158, aged 3 and over by
the age tables), so the 6,460,062 residents of other nationalities are not in it.

TABLES READ (1-based pages):
  Tableau 4.20, pp106-107   % by région: 13 named languages, "Aucune langue nationale parlée",
                            "Ensemble des autres langues nationales parlées", Total; 33 régions
                            (31 + the autonomous districts of Abidjan and Yamoussoukro) and
                            "Ensemble CI". Percentages to 1 dp.
  Annexe 20, pp140-141      counts nationally, by milieu: all 75 labels the census coded.
  Annexe 21, pp142-143      the same labels by age group: the second table, checked against 20.
  Annexe 25, p145           Ivorians by région and ethnic group: the région denominators.

THE BUILD. Tableau 4.20 is shares. Each région's count = share x its Ivorian population
(annex 25, all ages) x 21,295,158/22,840,168 (the language universe over all Ivorians: the
children under three), then each category is rescaled so its 33 régions sum to its national
count from annex 20 (religiondots' move for the same volume). "Ensemble des autres..." is
nationally the sum of the 61 other labels of annex 20, which is asserted.

THE data.gouv.ci COPY IS CORRUPT. data.gouv.ci's dataset "Répartition de la population
ivoirienne par la langue la plus parlée selon le milieu de résidence et le groupe d'âge"
(data354, origin ANStat) is Tableau 4.17 re-keyed with its columns shifted and digits changed
(Baoulé "988 959" in Abidjan where the tome prints 688 989; its 16.1% national Baoulé, which the
coverage sweep quoted, is the tome's 20.1% misread). Not used.

CHECKS (all must pass):
  1. the PDF is the 151-page volume with an %%EOF trailer
  2. Tableau 4.20: 33 régions and Ensemble CI, every row's 15 shares sum to its Total within
     0.8 (15 cells x 0.05), each région once
  3. annex 20: every row's Abidjan + other towns = urban, urban + rural = total; the labels sum
     to the printed Total, 21,295,158
  4. annex 21 agrees with annex 20 label by label within 10 people (it prints Baoulé 6 short)
  5. Ensemble CI's shares in Tableau 4.20 equal annex 20's counts over 21,295,158 to 1 dp
  6. annex 25: 33 régions, summing to 22,840,168 Ivorians
  7. the 33 régions join religiondots' ci_hexes units one to one (both ways)
  8. the rescale factors are printed; each must be within 0.85-1.15 (larger means the
     shares and the denominators disagree)
"""
import argparse
import re
import shutil
import sys
import unicodedata
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

RAW = HERE / "data" / "raw" / "ci"
OUT = HERE / "data" / "normalized" / "ci.csv"
PDF = RAW / "rgpg_tom1.pdf"
RD_PDF = HERE.parent / "religiondots" / "data" / "raw" / "ci" / "rgpg_tom1.pdf"
ORIGINAL = "https://www.anstat.ci/assets/publications/files/rgpg_tom1.pdf"
PDF_BYTES = 34_204_439
PAGES = 151

P_T420 = (105, 106)        # 0-based
P_A20 = (139, 140)
P_A21 = (141, 142)
P_A25 = (144,)

IVORIANS = 22_840_168      # all ages, annex 25 and annex 23
UNIVERSE = 21_295_158      # Ivorians aged 3+, the language tables' Total
OTHERS_PRINTED = 3_850_775  # Tableau 4.17's "Ensemble des autres langues"
A20_LABELS = 96            # 94 languages, "autre langue nationale à préciser", "aucune"

T420_COLUMNS = [
    "Baoulé", "Dioula", "Senoufo", "Malinké ou Malinka", "Agni", "Yacouba ou Dan", "Bété",
    "Akyé ou Attié", "Lobi", "Gouro", "Abbey", "Koulango", "Guéré",
    "Aucune langue nationale parlée", "Ensemble des autres langues nationales parlées",
]
NAMED = T420_COLUMNS[:13]
NONE_ = T420_COLUMNS[13]
OTHERS = T420_COLUMNS[14]

# Tableau 4.20 column -> annex 20 label (normalised)
A20_KEY = {
    "Baoulé": "baoule", "Dioula": "dioula", "Senoufo": "senoufo",
    "Malinké ou Malinka": "malinkeoumaninka", "Agni": "agni", "Yacouba ou Dan": "yacoubaoudan",
    "Bété": "bete", "Akyé ou Attié": "akyeouattie", "Lobi": "lobi", "Gouro": "gouro",
    "Abbey": "abbey", "Koulango": "koulango", "Guéré": "guere",
    "Aucune langue nationale parlée": "aucunelanguenationaleparlee",
}

# annex 25's names for the two districts; the rest fold to the same key as the hex layer's
A25_ALIAS = {"districtautonomedabidjan": "districtdabidjan",
             "districtautodeyakro": "districtdeyamoussoukro"}

SPACES = dict.fromkeys([0x00A0, 0x2007, 0x2008, 0x2009, 0x202F, 0x205F, 0x2000, 0x2001,
                        0x2002, 0x2003, 0x2004, 0x2005, 0x2006], " ")
PCT = re.compile(r"^\d{1,3},\d$")
NUM = re.compile(r"^\d+(?: \d+)*$")
JUNK = re.compile(r"^(Rapport thématique|Tableau|Source|selon |Langue parlée|Langues parlées|"
                  r"MILIEU DE RESIDENCE|Abidjan ville|Ensemble$|urbain$|Rural$|Groupe d|"
                  r"\d\d-\d\d ans|65 ans|\+$|Non$|spécifié$)")


def despace(s):
    return re.sub(r"\s+", " ", str(s).translate(SPACES)).strip()


def norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    if PDF.exists() and PDF.stat().st_size == PDF_BYTES:
        print("already have", PDF)
        return
    if RD_PDF.exists() and RD_PDF.stat().st_size == PDF_BYTES:
        shutil.copyfile(RD_PDF, PDF)
        print(f"copied religiondots' verified download -> {PDF}")
        return
    import json
    import urllib.parse
    import urllib.request
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
    target = ORIGINAL.split("://", 1)[1]
    cdx = ("https://web.archive.org/cdx/search/cdx"
           f"?url={urllib.parse.quote(target, safe='')}"
           "&output=json&fl=timestamp,statuscode,length&limit=300")
    with urllib.request.urlopen(urllib.request.Request(cdx, headers=ua), timeout=300) as r:
        rows = json.load(r)
    caps = [(int(x[2]), x[0]) for x in rows[1:] if x[1] == "200" and str(x[2]).isdigit()]
    length, ts = max(caps)
    url = f"https://web.archive.org/web/{ts}id_/{ORIGINAL}"
    print(f"{len(caps)} captures; taking {ts} (stored {length:,} bytes)")
    with urllib.request.urlopen(urllib.request.Request(url, headers=ua), timeout=1800) as r:
        body = r.read()
    if body[:4] != b"%PDF" or len(body) == 1 << 20 or not body.rstrip().endswith(b"%%EOF"):
        raise SystemExit(f"bad or truncated PDF ({len(body):,} bytes)")
    PDF.write_bytes(body)
    print(f"wrote {PDF} ({len(body):,} bytes)")


def lines_of(doc, pno):
    out = [despace(x) for x in doc.load_page(pno).get_text().splitlines()]
    out = [x for x in out if x]
    if out and out[0] == str(pno + 1):          # the page number
        out = out[1:]
    return out


def read_t420(doc):
    rows = []
    for pno in P_T420:
        ls = lines_of(doc, pno)
        i = 0
        while i < len(ls):
            if PCT.match(ls[i]):
                i += 1
                continue
            j, nums = i + 1, []
            while j < len(ls) and PCT.match(ls[j]):
                nums.append(float(ls[j].replace(",", ".")))
                j += 1
            if len(nums) == len(T420_COLUMNS) + 1:
                rows.append((ls[i], nums))
                i = j
            else:
                i += 1
    return rows


def read_rows(doc, pages, n):
    """Rows of a name (one or more lines) followed by exactly n integers."""
    rows = []
    for pno in pages:
        name, nums = [], []
        for ln in lines_of(doc, pno):
            if NUM.match(ln):
                nums.append(int(ln.replace(" ", "")))
                continue
            if nums:
                rows.append((name, nums))
                name, nums = [], []
            if not JUNK.match(ln):
                name.append(ln)
        if nums:
            rows.append((name, nums))
    out = []
    for name, v in rows:
        if len(name) > 1 and name[0] == "Total":     # the column header's last cell
            name = name[1:]
        if len(v) == n and name:
            out.append((" ".join(name), v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    import fitz
    import geopandas as gpd

    raw = PDF.read_bytes()
    assert raw.rstrip().endswith(b"%%EOF"), "no %%EOF trailer"
    doc = fitz.open(PDF)
    assert doc.page_count == PAGES, doc.page_count
    print(f"1. {PDF.name}: {doc.page_count} pages, {len(raw):,} bytes, %%EOF ok")

    # ---- 2. Tableau 4.20
    t420 = read_t420(doc)
    names = [n for n, _ in t420]
    assert len(t420) == 34 and names[-1] == "Ensemble CI", names
    assert len(set(map(norm, names))) == 34, "a région printed twice"
    worst = max(abs(sum(v[:-1]) - v[-1]) for _, v in t420)
    assert worst <= 0.8, worst
    print(f"2. Tableau 4.20: 33 régions + Ensemble CI; worst row sum off its Total by {worst:.1f}")
    shares = {n: dict(zip(T420_COLUMNS, v[:-1])) for n, v in t420}
    ens = shares.pop("Ensemble CI")

    # ---- 3. annex 20
    a20 = read_rows(doc, P_A20, 5)
    tot = [r for r in a20 if norm(r[0]) == "total"]
    a20 = [r for r in a20 if norm(r[0]) != "total"]
    assert len(tot) == 1 and tot[0][1][4] == UNIVERSE, tot
    # rows add up to within 2: the published cells are rounded (weighted) figures
    off = 0
    for nm, (ab, av, ur, ru, to) in a20:
        e = max(abs(ab + av - ur), abs(ur + ru - to))
        assert e <= 2, (nm, ab, av, ur, ru, to)
        off += e > 0
    nat = {norm(nm): v[4] for nm, v in a20}
    assert len(nat) == len(a20) == A20_LABELS, (len(nat), len(a20), [n for n, _ in a20])
    assert abs(sum(nat.values()) - UNIVERSE) <= 75, sum(nat.values())
    print(f"3. annex 20: {len(a20)} labels; rows add up within 2 ({off} off by 1-2, rounding); "
          f"labels sum to {sum(nat.values()):,} against the printed Total {UNIVERSE:,}")

    # ---- 4. annex 21
    a21 = {norm(nm): v[6] for nm, v in read_rows(doc, P_A21, 7) if norm(nm) != "total"}
    assert set(a21) == set(nat), set(a21) ^ set(nat)
    diff = {k: a21[k] - nat[k] for k in nat if a21[k] != nat[k]}
    assert all(abs(d) <= 10 for d in diff.values()), diff
    print(f"4. annex 21: same {len(a21)} labels; {len(diff)} differ, by at most "
          f"{max(map(abs, diff.values()), default=0)} ({diff})")

    # ---- national counts for the 15 drawn categories
    national = {c: nat[A20_KEY[c]] for c in NAMED + [NONE_]}
    rest = {k: v for k, v in nat.items() if k not in set(A20_KEY.values())}
    national[OTHERS] = sum(rest.values())
    assert abs(national[OTHERS] - OTHERS_PRINTED) <= 5, national[OTHERS]
    print(f"   the other {len(rest)} labels sum to {national[OTHERS]:,}; Tableau 4.17 prints "
          f"{OTHERS_PRINTED:,}")

    # ---- 5. Ensemble CI vs annex 20
    for c in T420_COLUMNS:
        pct = round(100 * national[c] / UNIVERSE, 1)
        assert abs(pct - ens[c]) <= 0.051, (c, pct, ens[c])
    print("5. Tableau 4.20's Ensemble CI row = annex 20 / 21,295,158 for all 15 categories")

    # ---- 6. annex 25
    a25 = read_rows(doc, P_A25, 8)
    hexes = gpd.read_file(RD_GEO / "ci" / "ci_hexes.gpkg", columns=["unit"], ignore_geometry=True)
    units = sorted(hexes["unit"].unique())
    ukey = {norm(u): u for u in units}
    assert len(ukey) == 33
    ivpop = {}
    for nm, v in a25:
        k = norm(nm)
        hits = [uk for uk in ukey if k.endswith(uk)] + \
               [ukey_ for ak, ukey_ in A25_ALIAS.items() if k.endswith(ak)]
        hits = sorted(set(hits))
        if len(hits) == 1:
            u = ukey[hits[0]]
            assert u not in ivpop, ("twice", u)
            assert abs(v[7] - sum(v[:7])) <= 3, (nm, v)     # rounded cells
            ivpop[u] = v[7]
        elif hits:
            raise SystemExit(f"annex 25 row {nm!r} matches {hits}")
    assert len(ivpop) == 33, sorted(set(units) - set(ivpop))
    assert abs(sum(ivpop.values()) - IVORIANS) <= 5, sum(ivpop.values())
    print(f"6. annex 25: 33 régions, {sum(ivpop.values()):,} Ivorians")

    # ---- 7. Tableau 4.20 names -> hex units
    t2u = {}
    for n in shares:
        k = norm(n)
        assert k in ukey, n
        t2u[n] = ukey[k]
    assert sorted(t2u.values()) == units
    print("7. Tableau 4.20's 33 régions = religiondots' 33 ci_hexes units, one to one")

    # ---- the counts
    k3 = UNIVERSE / IVORIANS
    rows = []
    for n, sh in shares.items():
        u = t2u[n]
        for c in T420_COLUMNS:
            rows.append(dict(geo_id=u, geo_level="region", geo_name=n, source_category=c,
                             pct=sh[c], raw=sh[c] / 100 * ivpop[u] * k3,
                             ivorians=ivpop[u]))
    df = pd.DataFrame(rows)
    fac = {c: national[c] / df.loc[df.source_category == c, "raw"].sum() for c in T420_COLUMNS}
    print("8. rescale factors (national count / sum of share x denominator):")
    for c in T420_COLUMNS:
        print(f"     {c:<50} x{fac[c]:.4f}")
    bad = {c: f for c, f in fac.items() if not 0.85 <= f <= 1.15}
    assert not bad, bad
    df["count"] = df["raw"] * df["source_category"].map(fac)
    df["count"] = df["count"].round().astype(int)
    df = df.drop(columns="raw")
    natrows = pd.DataFrame([dict(geo_id="CI", geo_level="national", geo_name="Côte d'Ivoire",
                                 source_category=nm, pct=round(100 * v[4] / UNIVERSE, 3),
                                 ivorians=IVORIANS, count=v[4]) for nm, v in a20])
    out = pd.concat([df, natrows], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    reg = df.groupby("source_category")["count"].sum()
    print(f"wrote {OUT}: {len(df):,} région rows ({(df['count'] > 0).sum():,} non-zero), "
          f"{len(natrows)} national rows; régions sum to {reg.sum():,}")


if __name__ == "__main__":
    main()
