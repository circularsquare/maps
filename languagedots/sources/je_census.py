"""Jersey: first language from the 2021 census's ethnic and cultural identity, with country of
birth for the groups that name no country -> data/normalized/je.csv.

    python sources/je_census.py [--fetch]

NO CENSUS LANGUAGE QUESTION (2021; the Chief Minister's written answer WQ.28/2024 says so for
Portuguese). Built as Bermuda (sources/bm.md): English for the island's own and British-Irish,
immigrant languages through sources/origin_mix.py (dest "je"). Every row `derived`.

THE TABLES, all Statistics Jersey's 2021 Census on opendata.gov.je (open data, Government of
Jersey), saved to data/raw/je/ (--fetch):
  ethnicity-2021-census.csv         one answer per person, 19 groups, sums to 103,267
  place-of-birth-2021-census.csv    8 groups, sums to 103,267
  2021-census-otherplaceofbirth.csv countries of birth with 10 or more residents (84 rows), the
                                    detail behind "Other European country" and "Elsewhere"
  2021-census-placeofbirth-age.csv  the 8 groups by age (rounded to 10)

WHY IDENTITY AND NOT BIRTHPLACE FOR THE MAIN GROUPS. Birthplace misses the Jersey-born children of
Portuguese and Polish parents: of 8,280 Portugal/Madeira-born residents only about 450 are under
20, but Portuguese-speaking homes are about 10% of school students (WQ.28/2024), so the second
generation is large. Identity (Portuguese / Madeiran 9,739) carries them. The identity table
partitions the census, so every resident is counted once.

THE MAPPING OF EACH IDENTITY GROUP
  Jersey, British, Irish, South African   English (as Bermuda/Barbados: the island-born and the
                                          British-Irish; South Africans in Jersey are English
                                          speakers, no Afrikaans share is known)
  French, Polish, Portuguese / Madeiran,
  Romanian                                that country's mix (origin_mix)
  White Other                             the 'other' birthplaces in Europe (detail table, less
                                          Romania, Poland, Portugal, France, Ireland, the UK) and
                                          English-speaking countries, weighted by head count
  Indian, Thai, Chinese                   that country's mix
  Asian Other                             Asian birthplaces in the detail table, weighted
  Black African, Black Caribbean          African / Caribbean birthplaces in the detail table, weighted
  Black Other, Mixed (3 groups)           English
Jersey's own language, Jerriais, is not drawn: it has no census count (the 2001 census found 113
people using it every day), under a tenth of a dot.

CHECKS (asserted): both tables sum to 103,267; each of Portugal/Madeira, Poland, Romania births is
below its identity group; the detail table is a subset of "Other European country" + "Elsewhere";
rows sum to the total.
"""
import argparse
import csv
import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
RAW = HERE / "data" / "raw" / "je"
OUT = HERE / "data" / "normalized" / "je.csv"
BASE = "https://opendata.gov.je/dataset/5840dfa1-8f67-4455-af42-241586ac8999/resource/"
FILES = {
    "ethnicity-2021-census.csv": "b4c15e10-5028-433b-acfa-224675b9b0f8/download/ethnicity-2021-census.csv",
    "place-of-birth-2021-census.csv": "a61ae77e-6a0d-44b4-9c1a-f5acd855e981/download/place-of-birth-2021-census.csv",
    "2021-census-otherplaceofbirth.csv": "a8778301-800a-4313-8db4-3711c86af3b8/download/2021-census-otherplaceofbirth.csv",
    "2021-census-placeofbirth-age.csv": "891b5c82-f161-4819-81ae-7917fd59ca87/download/2021-census-placeofbirth-age.csv",
}
TOTAL = 103_267
EN = "indoeuropean.germanic.english"

ISO = {
    "Angola": "AO", "Australia": "AU", "Austria": "AT", "Bangladesh": "BD", "Bahamas": "BS",
    "Barbados": "BB", "Belgium": "BE", "Botswana": "BW", "Bulgaria": "BG", "Cambodia": "KH",
    "Canada": "CA", "Cayman Islands": "KY", "Croatia": "HR", "Cyprus": "CY",
    "Czech Republic": "CZ", "Denmark": "DK", "Estonia": "EE", "Finland": "FI", "Germany": "DE",
    "Ghana": "GH", "Greece": "GR", "Hong Kong": "HK", "Hungary": "HU", "Indonesia": "ID",
    "Italy": "IT", "Jamaica": "JM", "Japan": "JP", "Kenya": "KE", "Latvia": "LV",
    "Lithuania": "LT", "Luxembourg": "LU", "Malawi": "MW", "Malaysia": "MY", "Malta": "MT",
    "Moldova": "MD", "Mozambique": "MZ", "Netherlands": "NL", "New Zealand": "NZ",
    "Nigeria": "NG", "Norway": "NO", "Pakistan": "PK", "Philippines": "PH", "Russia": "RU",
    "Singapore": "SG", "Slovakia": "SK", "South Korea": "KR", "Spain": "ES", "Sri Lanka": "LK",
    "Sweden": "SE", "Switzerland": "CH", "Taiwan": "TW", "Tanzania": "TZ",
    "Trinidad and Tobago": "TT", "Uganda": "UG", "Ukraine": "UA",
    "United States of America": "US", "Vietnam": "VN", "Zambia": "ZM", "Zimbabwe": "ZW",
}
ENGLISH_BORN = {"Australia", "Canada", "New Zealand", "United States of America"}   # drawn as English
WHITE_OTHER = ["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark",
               "Estonia", "Finland", "Germany", "Greece", "Hungary", "Italy", "Latvia", "Lithuania",
               "Luxembourg", "Malta", "Moldova", "Netherlands", "Norway", "Russia", "Slovakia",
               "Spain", "Sweden", "Switzerland", "Ukraine"] + sorted(ENGLISH_BORN)
ASIAN_OTHER = ["Philippines", "Hong Kong", "Pakistan", "Bangladesh", "Malaysia", "Singapore", "Sri Lanka",
               "Japan", "Indonesia", "Vietnam", "South Korea", "Taiwan", "Cambodia"]
BLACK_AFRICAN = ["Zimbabwe", "Kenya", "Nigeria", "Uganda", "Zambia", "Mozambique", "Malawi", "Ghana",
                 "Tanzania", "Botswana", "Angola"]
BLACK_CARIBBEAN = ["Jamaica", "Trinidad and Tobago", "Bahamas", "Barbados", "Cayman Islands"]
EN_GROUPS = ["Jersey", "British", "Irish", "South African"]
OWN = {"French": "FR", "Polish": "PL", "Portuguese / Madeiran": "PT", "Romanian": "RO",
       "Indian": "IN", "Thai": "TH", "Chinese": "CN"}


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, path in FILES.items():
        p = RAW / name
        if p.exists() and p.stat().st_size > 0:
            continue
        print(f"  fetching {BASE + path}")
        req = urllib.request.Request(BASE + path, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        p.write_bytes(data)


def read(name):
    with open(RAW / name, encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    if ap.parse_args().fetch:
        fetch()
    from terr_common import mix_written as mix

    eth = {(r[0], r[1]): int(r[2]) for r in read("ethnicity-2021-census.csv")[1:]}
    assert len(eth) == 19 and sum(eth.values()) == TOTAL, sum(eth.values())
    pob = {r[0]: int(r[1]) for r in read("place-of-birth-2021-census.csv")[1:]}
    tot = pob.pop("Total")
    assert tot == TOTAL and sum(pob.values()) == TOTAL, (tot, sum(pob.values()))
    detail = {r[0].strip(): int(r[1]) for r in read("2021-census-otherplaceofbirth.csv")[1:]}
    assert sum(detail.values()) <= pob["Other European country"] + pob["Elsewhere in the world"]
    # identity groups cover the births they imply
    for (g, e), b in (((("White", "Portuguese / Madeiran")), "Portugal / Madeira"),
                      ((("White", "Polish")), "Poland"), ((("White", "Romanian")), "Romania")):
        born = detail["Romania"] if b == "Romania" else pob[b]
        assert eth[(g, e)] > born, (e, eth[(g, e)], born)

    out = []

    def put(label, node, n):
        out.append(dict(geo_id="JE", geo_name="Jersey", identity=label, source_category=node, count=n))

    def spread(label, n, countries):
        w = {c: detail[c] for c in countries}
        s = sum(w.values())
        for c, wt in w.items():
            if c in ENGLISH_BORN:
                put(label, EN, n * wt / s)
            else:
                for node, sh in mix(ISO[c], "je").items():
                    put(label, node, n * wt / s * sh)

    for (g, e), n in eth.items():
        label = f"{g} {e}"
        if e in EN_GROUPS or g == "Mixed" or (g, e) == ("Black", "Other"):
            put(label, EN, n)
        elif e in OWN:
            for node, sh in mix(OWN[e], "je").items():
                put(label, node, n * sh)
        elif (g, e) == ("White", "Other"):
            spread(label, n, WHITE_OTHER)
        elif (g, e) == ("Asian", "Other"):
            spread(label, n, ASIAN_OTHER)
        elif (g, e) == ("Black", "African"):
            spread(label, n, BLACK_AFRICAN)
        elif (g, e) == ("Black", "Caribbean"):
            spread(label, n, BLACK_CARIBBEAN)
        else:
            raise SystemExit(f"unmapped identity group {g} / {e}")
    df = pd.DataFrame(out)
    assert abs(df["count"].sum() - TOTAL) < 1e-6, df["count"].sum()
    df["geo_level"] = "country"
    df["tier"] = "derived"
    df["year"] = 2021
    df.to_csv(OUT, index=False, encoding="utf-8")
    top = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"wrote {OUT}: {len(df)} rows, {df['count'].sum():,.0f} people, {len(top)} nodes")
    print((top.head(14).round(0)).to_string())


if __name__ == "__main__":
    main()
