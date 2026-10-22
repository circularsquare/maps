"""Guernsey: first language from the Electronic Census's country of birth -> data/normalized/gg.csv.

    python sources/gg_census.py [--fetch]

NO LANGUAGE QUESTION AND NO TRADITIONAL CENSUS SINCE 2001; Guernsey counts its people from
administrative records (the Rolling Electronic Census). Built as Bermuda (sources/bm.md): the
Guernsey-born and the British-Irish on English, the foreign-born on their birth country's languages
through sources/origin_mix.py (dest "gg"). Every row `derived`.

THE TABLE. States of Guernsey, Data & Analysis, Guernsey Quarterly Population, Employment and
Earnings Bulletin, Supplementary Data (population up to December 2023), sheet "Country of Birth"
(gov.gg/CHttpHandler.ashx?id=183493 -> data/raw/gg/supplementary_183493.xlsx), column
Count_31.03.23: 68 country rows and "Country of birth unknown", total 64,091 on 31 March 2023.
Checked against the Annual Electronic Census Report 2023, Table 5.2.1 (Guernsey 33,955, other Crown
Dependencies 420, UK and ROI 15,107, Portugal 1,326, Latvia 887, Other Europe 1,699, Rest of
world 3,985, unknown 6,712), p. 16 of data/raw/gg/electronic_census_2023.pdf.

THE CALLS
  * Guernsey, Alderney, Sark, Jersey, Isle of Man, UK parts, Ireland, British Overseas Territories,
    the US, Canada, Australia, New Zealand and Other Australasia: English (as sources/bm.md: many of
    those born in the large English-speaking countries are islanders' children, and the home mixes'
    Spanish and French would put speakers of those into Guernsey).
  * Every other named country: its origin_mix (dest "gg"), so Portuguese for Portugal, Latvian and
    Russian for Latvia, and so on.
  * "Other Europe / Africa / Asia / North America / South America" are group rows with no country:
    drawn on `other`, 509 people.
  * 6,712 people (10.5%) have no recorded country of birth. They are shared across the others pro
    rata (every row scaled by 64,091 / 57,379), so the island is drawn whole; left out, English would
    be undercounted by a tenth. Table 5.2.2 shows the unknowns skew to 5 to 15 years on island, so
    they may include more recent arrivals than the known group; no figure says by how much.

CHECKS (asserted): the rows sum to 64,091; the report's seven groups are reproduced from the
spreadsheet's rows (Other Europe and Rest of world differ by 4 the other way, their sum agrees); the count is the same Count_31.03.23 column the report prints.
"""
import argparse
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
RAW = HERE / "data" / "raw" / "gg"
OUT = HERE / "data" / "normalized" / "gg.csv"
URL = "https://www.gov.gg/CHttpHandler.ashx?id=183493&p=0"
XLSX = RAW / "supplementary_183493.xlsx"
EN = "indoeuropean.germanic.english"
TOTAL = 64_091
ENGLISH = {"Guernsey", "Alderney", "Isle of Man", "Jersey", "Sark", "England", "Northern Ireland",
           "Republic of Ireland", "Scotland", "Wales", "United Kingdom (not specified)",
           "Other United Kingdom British Overseas Territories", "Australia", "New Zealand",
           "Other Australasia", "Canada", "United States"}
OTHER = {"Other Europe", "Other Africa", "Other Asia", "Other North America", "Other South America"}
ISO = {
    "Austria": "AT", "Belgium": "BE", "Bulgaria": "BG", "Czech Republic": "CZ", "Denmark": "DK",
    "Finland": "FI", "France": "FR", "Germany": "DE", "Greece": "GR", "Hungary": "HU", "Italy": "IT",
    "Latvia": "LV", "Lithuania": "LT", "Malta": "MT", "Netherlands": "NL", "Poland": "PL",
    "Portugal": "PT", "Romania": "RO", "Russian Federation": "RU", "Slovakia": "SK", "Spain": "ES",
    "Sweden": "SE", "Serbia": "RS", "Switzerland": "CH", "Ukraine": "UA", "Egypt": "EG", "Ghana": "GH",
    "Kenya": "KE", "Mauritius": "MU", "Nigeria": "NG", "South Africa": "ZA", "Uganda": "UG",
    "Zambia": "ZM", "Zimbabwe": "ZW", "Bangladesh": "BD", "China": "CN", "India": "IN",
    "Malaysia": "MY", "Pakistan": "PK", "Philippines": "PH", "Singapore": "SG", "Thailand": "TH",
    "Jamaica": "JM", "Brazil": "BR", "Venezuela": "VE",
}
GROUPS = {   # the report's Table 5.2.1 rows from the spreadsheet's
    "Guernsey": 33_955, "Other Crown Dependencies": 420, "UK and ROI": 15_107, "Portugal": 1_326,
    "Latvia": 887, "Other Europe": 1_699, "Rest of world": 3_985, "unknown": 6_712,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    if ap.parse_args().fetch and not XLSX.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False                    # gov.gg serves an incomplete certificate chain
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        XLSX.write_bytes(urllib.request.urlopen(req, context=ctx, timeout=120).read())
    from terr_common import mix_written as mix

    d = pd.read_excel(XLSX, sheet_name="Country of Birth", header=1)
    d = d.rename(columns={"Count_31.03.23": "n"})[["Geographic_Area_Of_Birth", "Country_Of_Birth", "n"]]
    tot = int(d.loc[d["Country_Of_Birth"] == "TOTAL", "n"].iloc[0])
    unk = int(d.loc[d["Country_Of_Birth"] == "Country of birth unknown", "n"].iloc[0])
    rows = d[~d["Country_Of_Birth"].isin(["TOTAL", "Country of birth unknown"])].copy()
    assert tot == TOTAL and int(rows["n"].sum()) + unk == TOTAL, (tot, rows["n"].sum(), unk)
    g = rows.groupby("Geographic_Area_Of_Birth")["n"].sum()
    cd_other = int(rows[rows["Geographic_Area_Of_Birth"].eq("Crown Dependencies") &
                        ~rows["Country_Of_Birth"].eq("Guernsey")]["n"].sum())
    europe = int(rows[rows["Country_Of_Birth"].isin(["Portugal", "Latvia"])]["n"].sum())
    assert int(rows.loc[rows["Country_Of_Birth"] == "Guernsey", "n"].iloc[0]) == GROUPS["Guernsey"]
    assert cd_other == GROUPS["Other Crown Dependencies"], cd_other
    assert int(g["UK and ROI"]) == GROUPS["UK and ROI"], g["UK and ROI"]
    assert int(rows.loc[rows["Country_Of_Birth"] == "Portugal", "n"].iloc[0]) == GROUPS["Portugal"]
    assert int(rows.loc[rows["Country_Of_Birth"] == "Latvia", "n"].iloc[0]) == GROUPS["Latvia"]
    # the report's "Other Europe" and "Rest of world" differ from the spreadsheet's Europe rows less
    # Portugal and Latvia (1,699 against 1,703) and Rest of World rows (3,985 against 3,981) by 4
    # people the other way; their sum agrees (5,684). A country filed on the other side of the
    # line, kept as the spreadsheet has it
    assert abs(int(g["Europe"]) - europe - GROUPS["Other Europe"]) <= 5, int(g["Europe"]) - europe
    assert int(g["Europe"]) - europe + int(g["Rest of World"]) == GROUPS["Other Europe"] + GROUPS["Rest of world"]
    assert unk == GROUPS["unknown"]
    scale = TOTAL / (TOTAL - unk)

    out = []
    for _, r in rows.iterrows():
        c, n = r["Country_Of_Birth"], float(r["n"]) * scale
        if c in ENGLISH:
            m = {EN: 1.0}
        elif c in OTHER:
            m = {"other": 1.0}
        else:
            m = mix(ISO[c], "gg")
        for node, s in m.items():
            out.append(dict(geo_id="GG", geo_name="Guernsey", birthplace=c, source_category=node, count=n * s))
    df = pd.DataFrame(out)
    assert abs(df["count"].sum() - TOTAL) < 1e-6, df["count"].sum()
    df["geo_level"] = "island"
    df["tier"] = "derived"
    df["year"] = 2023
    df.to_csv(OUT, index=False, encoding="utf-8")
    top = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"wrote {OUT}: {len(df)} rows, {df['count'].sum():,.0f} people, {len(top)} nodes; "
          f"{unk:,} unknown birthplaces shared pro rata (x{scale:.4f})")
    print(top.head(10).round(0).to_string())


if __name__ == "__main__":
    main()
