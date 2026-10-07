"""Marshall Islands: 2021 Census of Population and Housing, Analytical Report, Table 3.3
(p. 15), national -> data/normalized/mh.csv.

    python sources/mh_census.py

The census asked which languages each person aged 5+ speaks (several allowed): 96.0% speak
Marshallese, 23.9% speak another language (mostly English learnt at school, AGENT_BRIEF section 2: not
a home language). No table names the other languages. So: Marshallese = 96.0% of the whole
enumerated population (42,418, Table 2.1; the 5+ rate applied to all ages), the 4.0% who do
not speak Marshallese on `other` (language not named; they are mostly the 6.8% non-citizens of
Table 3.4: Filipino, I-Kiribati, Chinese and US residents, but no table says which).
Report: https://www.infomarshallislands.com/wp-content/uploads/2025/12/Marshall-Islands-Census-2021.pdf
(data/raw/mh/rmi_census_2021.pdf).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "normalized" / "mh.csv"
TOTAL = 42_418
MARSHALLESE_SHARE = 0.960


def main():
    m = round(TOTAL * MARSHALLESE_SHARE)
    rows = [("Marshallese", m), ("Does not speak Marshallese", TOTAL - m)]
    df = pd.DataFrame([dict(geo_id="MH", geo_level="country", geo_name="Marshall Islands",
                            source_category=c, count=n, tier="derived", year=2021,
                            source_id="rmi_census_2021_t3.3") for c, n in rows])
    assert df["count"].sum() == TOTAL
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT.name}: {df.to_dict('records')}")


if __name__ == "__main__":
    main()
