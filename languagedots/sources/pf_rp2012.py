"""French Polynesia: Recensement de la population 2012 (ISPF / INSEE), language most often spoken
in the family, population aged 15 and over -> data/normalized/pf.csv.

    python sources/pf_rp2012.py [--fetch]

WHY 2012. The question ("langue la plus couramment parlée en famille", one answer, everyone 15+)
was asked in 2002, 2007, 2012, 2017 and 2022. Nothing below the territory is open for 2017 or
2022: their commune tables lived in ISPF's ASP.NET pivot tool (gone in 2022; the Wayback Machine
holds only its default view, French ability by subdivision), and 2022's results so far are a
6-page note, a workbook without a language sheet, and choropleth maps in five classes
(sources/pf.md §1). 2012's standard tables are the newest with any geography in them.

THE TABLE. Tableaux_standards_RP2012_Langues_v2003.xls (ISPF, archived by the Wayback Machine on
2014-10-09; ISPF's new site no longer serves it):
  - sheet LAN1b: 15+ by language group (Français, Langue polynésienne, Langue asiatique, Langue
    européenne (sauf français), Autres) x the 5 subdivisions x 10-year age band -> the counts;
  - sheet "Chiffres clés": the national figure for every language label (Tahitien 46,759,
    Marquisien 5,137, Hakka 665 ...) -> geo_level "national", kept for the record and the checks.
The groups are the finest split published by subdivision.

PLACEMENT SHARES (not counts). LAN_Atlas_quartier_indic.xls (ISPF's 2007 census atlas, Wayback
2021-08-24): the share of 15+ speaking a Polynesian language / French in the family for each of
165 "quartiers", which are the communes associées outside urban Tahiti and lettered sub-districts
inside it. Written to data/normalized/pf_place2007.csv per commune associée (lettered quartiers
averaged, unweighted: the file gives no populations). countries/pf.py uses them only to place
each subdivision's French and Polynesian dots inside it.

CHECKS (stop unless they hold): every subdivision's groups sum to its total, and its age bands to
it; the subdivisions sum to the territory in every group; LAN3b (knowledge of languages, same
census) gives the same 15+ total per subdivision; the national labels sum to their groups and the
groups to 202,825; the 2007 quartier codes all land on a commune associée of the boundary layer.
"""
import sys
import urllib.request
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "pf"
NORM = HERE / "data" / "normalized"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
FILES = {
    "Tableaux_standards_RP2012_Langues_v2003.xls":
        "http://web.archive.org/web/20141009011050id_/http://www.ispf.pf/docs/default-source/"
        "rp2012/Tableaux_standards_RP2012_Langues_v2003.xls?sfvrsn=0",
    "LAN_Atlas_quartier_indic.xls":
        "http://web.archive.org/web/20210824145657id_/https://www.ispf.pf/docs/default-source/"
        "cartographie/LAN_Atlas_quartier_indic.xls?sfvrsn=0",
}
SUBS = {"Iles Du Vent": "1", "Iles Sous-Le-Vent": "2", "Marquises": "3", "Australes": "4",
        "Tuamotu-Gambier": "5"}
GROUPS = ["Français", "Langue polynésienne", "Langue asiatique",
          "Langue européenne (sauf français)", "Autres"]
NATIONAL_15 = 202825
# "Chiffres clés": group header -> the labels under it (the sheet indents nothing, so by order)
KEY_GROUPS = {
    "Langues polynésiennes": 5, "Autres langues régionales françaises": 5,
    "Langues du Pacifique": 4, "Langues asiatiques": 4, "Langues européennes": 6,
    "Autres langues étrangères": 5,
}


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    import time
    for name, url in FILES.items():
        for attempt in range(4):        # the Wayback Machine answers 429 to quick requests
            try:
                data = urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                              timeout=300).read()
                break
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == 3:
                    raise
                time.sleep(30 * (attempt + 1))
        if data[:8] != b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            raise SystemExit(f"pf: {name} is not an .xls ({data[:60]!r})")
        (RAW / name).write_bytes(data)
        print(f"  {name}: {len(data):,} bytes")


def clean(s):
    return " ".join(str(s).split())


def sheet(name):
    import xlrd
    wb = xlrd.open_workbook(RAW / "Tableaux_standards_RP2012_Langues_v2003.xls")
    sh = wb.sheet_by_name(name)
    return [[clean(v) if isinstance(v, str) else v for v in sh.row_values(r)]
            for r in range(sh.nrows)]


def by_sub(rows, first_col, ncols):
    """{block name: [total row values, age rows...]} for a table laid out Ensemble/sub then ages."""
    out, cur = {}, None
    for r in rows:
        lab = r[0]
        if lab in SUBS or lab == "Ensemble":
            cur = lab
            out[cur] = {"total": [float(x or 0) for x in r[first_col:first_col + ncols]],
                        "ages": []}
        elif cur and isinstance(lab, str) and ("ans" in lab):
            out[cur]["ages"].append([float(x or 0) for x in r[first_col:first_col + ncols]])
    return out


def main():
    if "--fetch" in sys.argv or not all((RAW / f).exists() for f in FILES):
        fetch()

    # LAN1b: subdivision x group
    rows = sheet("LAN1b")
    head = next(r for r in rows if r[0] == "Subdivision et âge")
    if head[1:7] != ["Ensemble"] + GROUPS:
        raise SystemExit(f"pf: LAN1b header changed: {head[1:7]}")
    t = by_sub(rows, 1, 6)
    if set(t) != set(SUBS) | {"Ensemble"}:
        raise SystemExit(f"pf: LAN1b blocks {sorted(t)}")
    for k, v in t.items():
        tot = v["total"]
        if abs(sum(tot[1:]) - tot[0]) > 0.5:
            raise SystemExit(f"pf: LAN1b {k}: groups {sum(tot[1:])} != total {tot[0]}")
        if len(v["ages"]) != 8:
            raise SystemExit(f"pf: LAN1b {k}: {len(v['ages'])} age bands")
        for i in range(6):
            if abs(sum(a[i] for a in v["ages"]) - tot[i]) > 0.5:
                raise SystemExit(f"pf: LAN1b {k}: age bands do not sum in column {i}")
    nat = t["Ensemble"]["total"]
    if nat[0] != NATIONAL_15:
        raise SystemExit(f"pf: LAN1b territory 15+ {nat[0]}, expected {NATIONAL_15}")
    for i in range(6):
        s = sum(t[k]["total"][i] for k in SUBS)
        if abs(s - nat[i]) > 0.5:
            raise SystemExit(f"pf: subdivisions sum to {s} in column {i}, territory {nat[i]}")
    print(f"  LAN1b: 5 subdivisions x 5 groups; groups and age bands sum to every total; "
          f"subdivisions sum to the territory's {int(nat[0]):,} in every group")

    # LAN3b, same census, another table: 15+ per subdivision
    t3 = by_sub(sheet("LAN3b"), 1, 1)
    bad = {k: (t3[k]["total"][0], t[k]["total"][0]) for k in t
           if abs(t3[k]["total"][0] - t[k]["total"][0]) > 0.5}
    if bad:
        raise SystemExit(f"pf: LAN3b disagrees on the 15+ total: {bad}")
    print("  LAN3b (knowledge of French and Polynesian) gives the same 15+ total in all 5 "
          "subdivisions")

    # national labels
    key = sheet("Chiffres Clés")
    i0 = next(i for i, r in enumerate(key) if r[0] == "Ensemble des personnes de 15 ans et plus")
    lines = []
    for r in key[i0 + 1:]:
        if not r[0] or r[0].startswith("Source"):
            break
        lines.append((r[0], float(r[1])))
    if lines[0][0] != "Français" or lines[0][1] != nat[1]:
        raise SystemExit(f"pf: key figures French {lines[0]} vs LAN1b {nat[1]}")
    nat_rows, i, total = [("Français", lines[0][1], "Français")], 1, lines[0][1]
    while i < len(lines):
        grp, gv = lines[i]
        if grp == "Sourd et muet":      # a top-level label of its own, no group above it
            nat_rows.append((grp, gv, grp))
            total += gv
            i += 1
            continue
        n = KEY_GROUPS.get(grp)
        if n is None:
            raise SystemExit(f"pf: unexpected key-figures group {grp!r}")
        kids = lines[i + 1:i + 1 + n]
        if abs(sum(v for _, v in kids) - gv) > 0.5:
            raise SystemExit(f"pf: {grp} {gv} != its labels {sum(v for _, v in kids)}")
        nat_rows += [(lab, v, grp) for lab, v in kids]
        total += gv
        i += 1 + n
    if abs(total - NATIONAL_15) > 0.5:
        raise SystemExit(f"pf: key figures sum to {total}, expected {NATIONAL_15}")
    kg = {g: sum(v for _, v, gg in nat_rows if gg == g) for g in KEY_GROUPS}
    # LAN1b's "Autres" = regional French + Pacific + other foreign (incl. sign) in the key figures
    checks = {"Langue polynésienne": kg["Langues polynésiennes"],
              "Langue asiatique": kg["Langues asiatiques"],
              "Langue européenne (sauf français)": kg["Langues européennes"],
              "Autres": kg["Autres langues régionales françaises"] + kg["Langues du Pacifique"]
              + kg["Autres langues étrangères"]
              + sum(v for lab, v, _ in nat_rows if lab == "Sourd et muet")}
    for g, v in checks.items():
        if abs(nat[1 + GROUPS.index(g)] - v) > 0.5:
            raise SystemExit(f"pf: LAN1b {g} {nat[1 + GROUPS.index(g)]} vs key figures {v}")
    print(f"  key figures: {len(nat_rows)} national labels; they sum to their groups, the groups "
          f"to {NATIONAL_15:,}, and each group to LAN1b's territory column")

    out = []
    for name, code in SUBS.items():
        for g, v in zip(GROUPS, t[name]["total"][1:]):
            out.append(dict(geo_id=code, geo_level="subdivision", geo_name=name,
                            source_category=g, count=int(v)))
    for lab, v, grp in nat_rows:
        out.append(dict(geo_id="PF", geo_level="national", geo_name="Polynésie française",
                        source_category=lab, count=int(v)))
    df = pd.DataFrame(out)
    df["year"] = 2012
    df["source_id"] = "pf_rp2012_lan"
    NORM.mkdir(parents=True, exist_ok=True)
    df.to_csv(NORM / "pf.csv", index=False, encoding="utf-8")
    sub = df[df["geo_level"] == "subdivision"].pivot(index="geo_name", columns="source_category",
                                                      values="count")[GROUPS]
    print(sub.to_string())
    print(f"  wrote {NORM / 'pf.csv'} ({len(df)} rows)")

    # 2007 placement shares per commune associée
    import xlrd
    sh = xlrd.open_workbook(RAW / "LAN_Atlas_quartier_indic.xls").sheet_by_index(0)
    q = []
    for r in range(3, sh.nrows):
        code = sh.cell_value(r, 0)
        if isinstance(code, float):
            code = str(int(code))
        code = str(code).strip()
        if not code or code.startswith("Total"):
            continue
        q.append((code, float(sh.cell_value(r, 1)), float(sh.cell_value(r, 2))))
    q = pd.DataFrame(q, columns=["quartier", "share_pol", "share_fr"])
    ca = pd.read_csv(RAW / "communesassociees.csv", sep=";", dtype=str)
    comas = set(ca["IDComas"])
    one = ca.groupby("IDCom")["IDComas"].apply(list).to_dict()

    def to_comas(code):
        if code in comas:
            return [code]
        if code[-1].isalpha():          # a lettered quartier: its commune (associée) code
            base = code[:-1]
            if base in comas:
                return [base]
            if base[:2] in one and len(base) == 2:
                return one[base[:2]]    # e.g. 52A-C -> Teva I Uta's 521 and 522
        raise SystemExit(f"pf: 2007 quartier {code} matches no commune associée")
    q["comas"] = q["quartier"].map(to_comas)
    q = q.explode("comas")
    sh2007 = q.groupby("comas")[["share_pol", "share_fr"]].mean()
    missing = comas - set(sh2007.index)
    if missing:
        raise SystemExit(f"pf: communes associées without a 2007 share: {sorted(missing)}")
    sh2007.reset_index().to_csv(NORM / "pf_place2007.csv", index=False)
    print(f"  2007 atlas: {len(q['quartier'].unique())} quartiers onto all {len(comas)} communes "
          f"associées; Polynesian share {sh2007['share_pol'].min():.2f}-"
          f"{sh2007['share_pol'].max():.2f} -> {NORM / 'pf_place2007.csv'}")


if __name__ == "__main__":
    main()
