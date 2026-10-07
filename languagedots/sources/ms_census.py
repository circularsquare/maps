"""Montserrat: first language from the 2011 census's place of birth, national
-> data/normalized/ms.csv.

    python sources/ms_census.py [--fetch]

NO CENSUS LANGUAGE QUESTION (2011). Built as St Kitts and Antigua (sources/kn.md, ag.md): the
Montserrat-born on the Leeward creole (Glottolog's Antigua and Barbuda Creole English, anti1245,
whose countries include MS; node `antiguan` from tree.d/bb.txt), the foreign-born on their birth
country's languages through sources/origin_mix.py (dest "ms"). Every row `derived`.

THE TABLE: ECLAC's REDATAM WebServer for the Montserrat 2011 PHC (prod.redatam.org/binmsr, base
PHC2011, open), Frequency of PERSON.Q47_BORN "47. Place of birth", saved as
data/raw/ms/ms_q47_born.htm. Q47 codes Montserrat-born people by village (27 + "Elsewhere in
Montserrat") and the foreign-born by country. Total 4,775 (persons in the base; the published
usual-resident count is 4,922). "Another country" (54) on `other`; "Don't Know" and "Not Stated"
(7) not drawn. The 2018 Labour Force base on the same server is a survey; the 2023 census has
no open tables yet.

CHECKS: the rows sum to the printed Total; every row is a village or a listed country.
"""
import html
import re
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sources"))
sys.path.insert(0, str(ROOT / "taxonomy"))
RAW = ROOT / "data" / "raw" / "ms" / "ms_q47_born.htm"
OUT = ROOT / "data" / "normalized" / "ms.csv"
CREOLE = "creole.english_based.antiguan"
COUNTRY = {
    "Antigua and Barbuda": "AG", "Barbados": "BB", "Canada": "CA", "Dominica": "DM",
    "Dominican Republic": "DO", "Grenada": "GD", "Guyana": "GY", "Haiti": "HT", "Jamaica": "JM",
    "Netherlands Antilles": "CW", "Saint Kitts and Nevis": "KN", "Saint Lucia": "LC",
    "Saint Vincent and the Grenadines": "VC", "Trinidad and Tobago": "TT",
    "United States of America": "US", "United States Virgin Islands": "VI", "Netherlands": "NL",
    "United Kingdom of Great Britain & Northern Ireland": "GB", "Sri Lanka": "LK", "India": "IN",
    "Nigeria": "NG",
}
NOT_DRAWN = {"Don’t Know", "Don't Know", "Not Stated"}
EN = "indoeuropean.germanic.english"
CR = "creole.english_based."
# birthplace -> node where origin_mix is not used (sources/kn_census.py's conventions): the
# English Caribbean's creoles by name (origins not drawn fall back to English otherwise), and the
# US, Canada and Britain as English (many are Montserratians' children born abroad)
NODE = {
    "Antigua and Barbuda": CREOLE, "Saint Kitts and Nevis": CREOLE,
    "Guyana": CR + "guyanese", "Grenada": CR + "grenadian", "Trinidad and Tobago": CR + "trinidadian",
    "Saint Lucia": "creole.french_based.antillean", "United States Virgin Islands": CR + "virgin_islands",
    "United States of America": EN, "Canada": EN,
    "United Kingdom of Great Britain & Northern Ireland": EN,
}


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings()
    r = requests.post("https://prod.redatam.org/binmsr/RpWebStats.exe/Frequency?", data={
        "MAIN": "WebServerMain.inl", "BASE": "PHC2011", "LANG": "ENG", "CODIGO": "XXUSUARIOXX",
        "ITEM": "FREQ1", "MODE": "RUN", "inputTitle": "", "ROW": "PERSON.Q47_BORN",
        "SELECTION": "ALL", "PERCENT": "OFF", "FORMAT": "HTML", "Submit": "Execute"},
        headers={"User-Agent": "Mozilla/5.0"}, timeout=600, verify=False)
    r.raise_for_status()
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text(r.text, encoding="utf-8")


def rows():
    body = RAW.read_text(encoding="utf-8")
    out = {}
    total = None
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S | re.I):
        c = [html.unescape(re.sub(r"<[^>]+>", "", x)).replace("\xa0", " ").strip()
             for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S | re.I)]
        c = [x for x in c if x]
        if len(c) == 4 and c[2].endswith("%") and c[0] != "47. Place of birth":
            n = int(re.sub(r"\D", "", c[1]))
            if c[0] == "Total":
                total = n
            else:
                out[c[0]] = n
    assert total and sum(out.values()) == total, (total, sum(out.values()))
    return out, total


def main():
    from origin_mix import mix
    r, total = rows()
    acc = {}
    native = foreign = 0
    for lab, n in r.items():
        if lab in NOT_DRAWN:
            continue
        if lab in NODE:
            m = {NODE[lab]: 1.0}
            foreign += n
        elif lab in COUNTRY:
            m = mix(COUNTRY[lab], "ms")
            foreign += n
        elif lab == "Another country":
            m = {"other": 1.0}
            foreign += n
        else:   # a Montserrat village, or "Elsewhere in Montserrat"
            m = {CREOLE: 1.0}
            native += n
        for node, s in m.items():
            acc[node] = acc.get(node, 0) + n * s
    drawn = native + foreign
    print(f"Q47: {total:,} people; Montserrat-born {native:,}, foreign-born {foreign:,}, "
          f"not drawn {total - drawn}")
    s = pd.Series(acc)
    fl = s.apply(int)
    fl[(s - fl).sort_values(ascending=False).index[:drawn - int(fl.sum())]] += 1
    df = pd.DataFrame([dict(geo_id="MS", geo_level="country", geo_name="Montserrat",
                            source_category=k, count=int(v), tier="derived", year=2011,
                            source_id="msr_phc2011_q47") for k, v in fl.items() if v > 0])
    assert df["count"].sum() == drawn
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(df.sort_values("count", ascending=False).head(12)[["source_category", "count"]]
          .to_string(index=False))


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    main()
