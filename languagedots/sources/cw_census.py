"""Curaçao, Census 2023 (Senso 2023, CBS Curaçao): language spoken most often at home
-> data/normalized/cw.csv.

    python sources/cw_census.py [--fetch]

THE TABLE. CBS publishes the 2023 home-language answers only as percentages for the whole island:
"Eerste Resultaten Census 2023" (5 June 2024), page 21, the box "Meest gesproken taal/talen thuis",
column "eerste taal": Papiamentu 78.0%, Spaans 8.4%, Nederlands 7.9%, Engels 3.8%, overig 2.0%,
of the 147,498 people who answered. No 2023 language table by geozone or neighbourhood exists: the
2023 tables (senso.cbs.cw), the neighbourhood viewer (cbs-curacao.github.io/ndv-static-site) and
CBS's ArcGIS layers carry no language variable (sources/cw.md §1). The counts here are those
shares times 147,498, scaled so they sum to it (the printed shares sum to 100.1%), so every row is
`derived`, good to about +-75 people.

THE CHECK. Census 2011's Table D-6 ("Most spoken language (in private households)", per
household, applied to its members) is fetched too and written as geo_level `national_2011`, with
its ten categories, for the record's comparison; countries/cw.py draws 2023 only. A second check
is the 2023 publication "Migranten in Curaçao" (Table 18, shares by country of birth), read by
hand in sources/cw.md.
"""
import csv
import re
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "cw"
OUT = HERE / "data" / "normalized" / "cw.csv"
PDF = RAW / "eerste_resultaten_census_2023.pdf"
D6 = RAW / "d-6-population-by-most-spoken-language-2011.xlsx"
URLS = {
    PDF: "https://cuatro.sim-cdn.nl/sensocbs/uploads/"
         "05062024_-_publicatie_eerste_resultaten_census_2023.pdf",
    D6: "https://cuatro.sim-cdn.nl/cbscuracao/uploads/d-6-population-by-most-spoken-language.xlsx",
}
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
LABELS = ["Papiamentu", "Spaans", "Nederlands", "Engels", "overig"]
ANSWERED = 147_498
POP_2023 = 155_826          # Census 2023 total population (Tables D-5, G-3)


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for path, url in URLS.items():
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        data = urllib.request.urlopen(req, timeout=300).read()
        path.write_bytes(data)
        print(f"  {path.name}: {len(data):,} bytes")


def read_2023():
    import fitz
    doc = fitz.open(PDF)
    if doc.page_count != 24:
        raise SystemExit(f"cw: the first-results PDF has {doc.page_count} pages, expected 24")
    t = doc[20].get_text()
    if "Meest gesproken taal/talen thuis" not in t:
        raise SystemExit("cw: page 21 no longer holds the home-language box")
    m = re.search(r"Van de\s*([\d.]+)\s*personen die deze vraag beantwoord", t)
    if not m or int(m.group(1).replace(".", "")) != ANSWERED:
        raise SystemExit(f"cw: answered count changed: {m and m.group(1)}")
    shares = {}
    for lab in LABELS:
        # the box prints label, then first / second / third language shares, one per line
        m = re.search(rf"\n{lab}\s*\n\s*([\d.]+)%\s*\n\s*([\d.]+)%\s*\n\s*([\d.]+)%", t)
        if not m:
            raise SystemExit(f"cw: no row for {lab!r} on page 21")
        shares[lab] = float(m.group(1))
    total = sum(shares.values())
    if abs(total - 100.0) > 0.25:
        raise SystemExit(f"cw: first-language shares sum to {total:.1f}%")
    # the prose repeats the four named shares; they must agree with the box
    for lab, word in [("Papiamentu", "78 procent"), ("Spaans", "8,4 procent"),
                      ("Nederlands", "7,9%"), ("Engels", "3.8%")]:
        if word not in t:
            raise SystemExit(f"cw: prose no longer says {word!r} for {lab}")
    return shares, total


def read_2011():
    import openpyxl
    ws = openpyxl.load_workbook(D6, data_only=True).worksheets[0]
    rows = [r for r in ws.iter_rows(values_only=True) if any(v is not None for v in r)]
    head = next(r for r in rows if r[1] == "Age Group")
    tot = next(r for r in rows if r[1] == "Total")
    cols = [c for c in head[2:] if c]
    vals = dict(zip(cols, tot[2:2 + len(cols)]))
    ages = [r for r in rows if r is not head and r is not tot and isinstance(r[2], (int, float))]
    for c in cols:
        if sum(r[2 + cols.index(c)] for r in ages) != vals[c]:
            raise SystemExit(f"cw: 2011 D-6 column {c} does not sum to its total")
    if sum(v for k, v in vals.items() if k != "Total") != vals["Total"] or vals["Total"] != 147_862:
        raise SystemExit("cw: 2011 D-6 categories do not sum to 147,862")
    return vals


def main():
    if "--fetch" in sys.argv or not PDF.exists() or not D6.exists():
        fetch()
    shares, total = read_2023()
    rows = []
    counts = {lab: shares[lab] / total * ANSWERED for lab in LABELS}
    for lab in LABELS:
        rows.append(dict(geo_id="CW", geo_level="national", geo_name="Curaçao",
                         source_category=lab, count=round(counts[lab], 1), tier="derived",
                         source_id="cbs_census2023_eerste_resultaten_p21", year=2023,
                         note=f"{shares[lab]}% of {ANSWERED:,} who answered"))
    v11 = read_2011()
    for lab, v in v11.items():
        if lab == "Total":
            continue
        rows.append(dict(geo_id="CW", geo_level="national_2011", geo_name="Curaçao",
                         source_category=lab, count=v, tier="measured",
                         source_id="cbs_census2011_table_d6", year=2011,
                         note="private households; asked per household"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  2023: shares sum to {total:.1f}%; {ANSWERED:,} answered of {POP_2023:,} "
          f"({1 - ANSWERED / POP_2023:.1%} not)")
    for lab in LABELS:
        print(f"    {lab:<11} {shares[lab]:>5.1f}%  {counts[lab]:>9,.0f}")
    t11 = v11["Total"] - v11["Not reported"]
    print(f"  2011 (D-6, per household, {t11:,} reported):")
    for lab, v in v11.items():
        if lab not in ("Total", "Not reported"):
            print(f"    {lab:<14} {v:>7,}  {v / t11:6.1%}")
    print(f"  wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
