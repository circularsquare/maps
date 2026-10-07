"""The Bahamas: first language from the 2010 census's citizenship and race tables per island,
as shares on the 2022 census island populations -> data/normalized/bs.csv.

    python sources/bs_census.py --fetch     # --fetch downloads the eighteen 2010 island reports
    python sources/bs_census.py

NO CENSUS LANGUAGE QUESTION (2010 and 2022 questionnaires; sources/bs.md). Built under the
2026-10-05 ruling for countries with no language question (AGENT_BRIEF §2): the national
language for citizens, immigrant languages proxied by citizenship. Every row `derived`.

THE TABLES. BNSI's 2010 Census island reports (stats.gov.bs, one PDF per island, all 18 census
islands), each with
  * Table 8.0, total population by racial group (BLACK, WHITE, BLACK AND WHITE, ...);
  * Table 9.0, total population by country of citizenship (BAHAMAS, HAITI, JAMAICA, ...).
Only the TOTAL line of each category is read (the MALE/FEMALE lines follow it).

THE MODEL, per island:
  * a foreign citizenship -> its country's language (taxonomy/bs2010.py), e.g. Haitian
    nationals -> Haitian Creole. Haitian nationals include the Bahamas-born children of Haitian
    parents, who are not citizens at birth.
  * Bahamian citizens -> Bahamian Creole, except white Bahamians -> English (White Bahamian
    English is an English dialect, not a creole). White Bahamian citizens are estimated as the
    island's WHITE count less its citizens of North America, Europe and Oceania (floored at 0).
  * not stated citizenship: left out of the shares (gap), so the shares are of stated answers.
The 2010 shares are applied to each island's 2022 census population (religiondots'
normalized bs.csv TOTAL rows, read only).

CHECKS: each island's citizenship rows sum to its Table 9.0 TOTAL and its race rows to its Table
8.0 TOTAL; the two totals agree; the 18 islands sum to the 2010 census total 351,461; the 2022
islands sum to 398,165; each island's drawn rows sum to its 2022 population.
"""
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

RAW = HERE / "data" / "raw" / "bs"
OUT = HERE / "data" / "normalized" / "bs.csv"
URL = "https://stats.gov.bs/wp-content/uploads/2020/08/{}-2010-CENSUS-REPORT.pdf"

# 2010 report file stem -> religiondots island id (BNSI's census island code) and name
ISLANDS = {
    "NEW-PROVIDENCE": ("01", "New Providence"), "GRAND-BAHAMA": ("02", "Grand Bahama"),
    "ABACO": ("03", "Abaco"), "ACKLINS": ("04", "Acklins"), "ANDROS": ("05", "Andros"),
    "BERRY-ISLANDS": ("06", "Berry Islands"), "BIMINIS": ("07", "Bimini"),
    "CAT-ISLAND": ("08", "Cat Island"), "CROOKED-ISLAND": ("09", "Crooked Island"),
    "ELEUTHERA": ("10", "Eleuthera"), "EXUMA-CAYS": ("11", "Exuma"),
    "HARBOUR-ISLAND": ("12", "Harbour Island"), "INAGUA": ("13", "Inagua"),
    "LONG-ISLAND": ("14", "Long Island"), "MAYAGUANA": ("15", "Mayaguana"),
    "RAGGED-ISLAND": ("16", "Ragged Island"), "SAN-SALVADOR": ("17", "San Salvador and Rum Cay"),
    "SPANISH-WELLS": ("18", "Spanish Wells"),
}
RACES = {"BLACK", "WHITE", "BLACK AND WHITE", "BLACK AND OTHER", "WHITE AND OTHER", "ASIAN",
         "EAST INDIAN", "OTHER RACES", "NOT STATED"}
# spelling variants in the reports -> one label (the same answer, misprinted)
ALIAS = {"JAMACIA": "JAMAICA", "TRINADAD AND TOBAGO": "TRINIDAD AND TOBAGO",
         "TURKS & CAICOS ISLANDS": "TURKS AND CAICOS ISLANDS", "U.S.A.": "U.S.A",
         "SW ITZERLAND": "SWITZERLAND", "OTHER COM M OM W EALTH": "OTHER COMMONWEALTH",
         "OTHER COMMOMWEALTH": "OTHER COMMONWEALTH",
         "NON-COM M ONW EALTH COUNTRIES": "NON-COMMONWEALTH COUNTRIES",
         "NON-COMONWEALTH COUNTRIES": "NON-COMMONWEALTH COUNTRIES"}
# citizenships whose holders the white-citizen estimate subtracts (North America, Europe, Oceania)
WESTERN = {"U.S.A", "CANADA", "BERMUDA", "UNITED KINGDOM", "IRELAND", "FRANCE", "GERMANY",
           "NETHERLANDS AND HOLLAND", "BELGIUM", "SWITZERLAND", "AUSTRIA", "ITALY", "SPAIN",
           "PORTUGAL", "GREECE", "SWEDEN", "NORWAY", "DENMARK", "POLAND", "ROMANIA", "RUSSIA",
           "UKRAINE", "BULGARIA", "ESTONIA", "AUSTRALIA", "NEW ZEALAND"}
TOTAL_2010, TOTAL_2022 = 351_461, 398_165
NUM = re.compile(r"^[\d,]+$")


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for stem in ISLANDS:
        p = RAW / f"bs_2010_{stem.lower()}.pdf"
        if not p.exists():
            req = urllib.request.Request(URL.format(stem),
                                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0)"})
            p.write_bytes(urllib.request.urlopen(req, timeout=120).read())
            print("fetched", p.name)


def _table(doc, num):
    """Lines of every page of Table <num>.0, in order."""
    out = []
    for pg in doc:
        lines = [x.strip() for x in pg.get_text().split("\n") if x.strip()]
        if len(lines) > 2 and re.match(rf"TABLE {num}\.0\b", lines[1]):
            out += lines[1:] + ["<page>"]    # drop the page number, mark the page break
    return out


def _totals(lines, wanted=None):
    """{label: TOTAL count} for each category line followed by a number. The first `TOTAL` is the
    island's; region headers (`**** X ****`, then `TOTAL`) are skipped; MALE/FEMALE ignored."""
    cats, island_total, seen_male = {}, None, False
    for i, (a, b) in enumerate(zip(lines, lines[1:])):
        if (b == "MALE" and a not in ("TOTAL", "MALE", "FEMALE") and i + 2 < len(lines)
                and NUM.match(lines[i + 2])):
            # a few rows come out of the PDF as `label, MALE, <total row>, FEMALE, <male row>,
            # <female row>` (Inagua's Canada and U.S.A): the number after MALE is the total
            b = lines[i + 2]
        if not NUM.match(b):
            continue
        n = int(b.replace(",", ""))
        if a == "TOTAL":
            if island_total is None:
                island_total = n
            continue
        if a in ("MALE", "FEMALE"):
            seen_male = True
            continue
        if a.startswith("*") or a.startswith("<") or not re.search(r"[A-Z]", a):
            continue
        a = ALIAS.get(a, a)
        if wanted is not None:
            if a not in wanted:
                continue
            if a in cats:          # race table: the MALE and FEMALE blocks repeat the labels
                continue
        elif a in cats:
            raise SystemExit(f"repeated citizenship {a!r}")
        cats[a] = n
    return island_total, cats


def parse():
    import fitz
    rows = []
    for stem, (gid, name) in ISLANDS.items():
        doc = fitz.open(RAW / f"bs_2010_{stem.lower()}.pdf")
        rt, race = _totals(_table(doc, 8), RACES)
        ct, cit = _totals(_table(doc, 9))
        assert rt == ct, (stem, rt, ct)
        assert sum(race.values()) == rt, (stem, race, rt)
        assert sum(cit.values()) == ct, (stem, sum(cit.values()), ct)
        assert "BAHAMAS" in cit, stem
        rows.append(dict(geo_id=gid, name=name, total=ct, race=race, cit=cit))
    tot = sum(r["total"] for r in rows)
    assert tot == TOTAL_2010, tot
    return rows


def build(rows):
    import bs2010   # noqa: F401  (labels are checked against the mapping here)
    rd = pd.read_csv(RD / "data" / "normalized" / "bs.csv", dtype={"geo_id": str},
                     keep_default_na=False, na_values=[])
    rd = rd[(rd["geo_level"] == "island") & (rd["source_category"] == "TOTAL")]
    pop22 = dict(zip(rd["geo_id"], pd.to_numeric(rd["count"])))
    assert len(pop22) == 18 and sum(pop22.values()) == TOTAL_2022, sum(pop22.values())
    names22 = dict(zip(rd["geo_id"], rd["geo_name"]))
    out = []
    nat = {}
    for r in rows:
        assert names22[r["geo_id"]] == r["name"], (r["geo_id"], names22[r["geo_id"]], r["name"])
        cit = dict(r["cit"])
        cit.pop("NOT STATED", None)
        bah = cit.pop("BAHAMAS")
        western = sum(v for k, v in cit.items() if k in WESTERN)
        white_cit = min(bah, max(0, r["race"].get("WHITE", 0) - western))
        parts = {"BAHAMAS (white)": white_cit, "BAHAMAS (not white)": bah - white_cit}
        parts.update(cit)
        stated = sum(parts.values())
        p22 = pop22[r["geo_id"]]
        # largest-remainder rounding so each island sums to its 2022 population exactly
        raw = {k: p22 * v / stated for k, v in parts.items() if v > 0}
        cnt = {k: int(v) for k, v in raw.items()}
        for k in sorted(raw, key=lambda k: raw[k] - cnt[k], reverse=True)[:p22 - sum(cnt.values())]:
            cnt[k] += 1
        assert sum(cnt.values()) == p22
        for k, v in cnt.items():
            nat[k] = nat.get(k, 0) + parts[k]
            if v == 0:
                continue
            bs2010.resolve(k)
            out.append(dict(geo_id=r["geo_id"], geo_level="island", geo_name=r["name"],
                            source_category=k, count=v, tier="derived", year=2022,
                            note=f"2010 census: {parts[k]} of {stated} stated on the island"))
        print(f"{r['geo_id']} {r['name']:<26} 2010 {r['total']:>7,}  2022 {p22:>7,}  "
              f"white {r['race'].get('WHITE', 0):>6,}  white citizens {white_cit:>6,}  "
              f"Haitian {cit.get('HAITI', 0):>6,}")
    df = pd.DataFrame(out)
    assert df["count"].sum() == TOTAL_2022
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(df)} rows, {df['geo_id'].nunique()} islands, "
          f"{df['count'].sum():,} people")
    s = sum(nat.values())
    for k, v in sorted(nat.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  2010 national {k:<28} {v:>8,}  {v / s:6.2%}")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE / "taxonomy"))
    if "--fetch" in sys.argv:
        fetch()
    build(parse())
