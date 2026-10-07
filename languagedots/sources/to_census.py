"""Tonga: language used at home, 2021 census, by division, placed on village populations
-> data/normalized/to.csv.

    python sources/to_census.py

THE QUESTION (2021 questionnaire ID7.7, p.350 of the report): "What language does this person
speak at home?" 1 Tongan language only / 2 Tongan and other language(s) / 3 Tongan language is
not used at home. Asked of everyone 5 and over.

THE TABLE: Tonga Statistics Department, *Tonga 2021 Census of Population and Housing, Volume 1:
Basic Tables* (data/raw/to/census_report_vol1_2021.pdf, from tongastats.gov.to), Table G 48 (pp.
155-158), language use at home by age, sex and division. Division totals typed in below (both
sexes); the script asserts they sum to the TONGA row.

THE MAPPING: "Tongan only" and "Tongan and other" -> Tongan (a person who speaks Tongan and
another language at home is drawn on Tongan, the one language named; the other is not named);
"not used at home" -> `other` (the language is not recorded). Each division's three shares (of
the population 5+) are applied to the 2021 village populations religiondots parsed from the
same census (its normalized to.csv, every religion row incl. refusals, read only), so under-5s
take their division's shares and the dots follow villages within each division. Rows `derived`.

CHECKS: the five divisions sum to Table G 48's TONGA row in all four columns; every village has a
division; every village's rows sum to its population.
"""
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

OUT = HERE / "data" / "normalized" / "to.csv"
# Table G 48, both sexes, population 5+: (total, Tongan only, Tongan and other, Tongan not used)
G48 = {
    "Tongatapu": (66_295, 55_105, 10_298, 892),
    "Vava'u": (12_550, 11_343, 1_133, 74),
    "Ha'apai": (5_027, 4_677, 327, 23),
    "'Eua": (4_342, 3_721, 605, 16),
    "Ongo Niua": (1_040, 982, 57, 1),
}
TONGA = (89_254, 75_828, 12_420, 1_006)
LABELS = ("Tongan language only", "Tongan and other language(s)", "Tongan language is not used at home")


def main():
    for j in range(4):
        assert sum(v[j] for v in G48.values()) == TONGA[j], j
    for k, v in G48.items():
        assert v[0] == sum(v[1:]), k
    rd = pd.read_csv(RD / "data" / "normalized" / "to.csv", keep_default_na=False, na_values=[""])
    rd["division"] = rd["note"].str.split("|").str[0]
    vil = rd.groupby(["geo_id", "geo_name", "division"], as_index=False)["count"].sum()
    assert vil["geo_id"].is_unique
    assert set(vil["division"]) == set(G48), set(vil["division"]) ^ set(G48)
    out = []
    for _, r in vil.iterrows():
        t, *parts = G48[r["division"]]
        raw = [r["count"] * p / t for p in parts]
        cnt = [int(x) for x in raw]
        for i in sorted(range(3), key=lambda i: raw[i] - cnt[i], reverse=True)[:r["count"] - sum(cnt)]:
            cnt[i] += 1
        assert sum(cnt) == r["count"]
        for lab, n in zip(LABELS, cnt):
            if n:
                out.append(dict(geo_id=r["geo_id"], geo_level="village", geo_name=r["geo_name"],
                                source_category=lab, count=n, tier="derived", year=2021,
                                note=f"{r['division']} shares, Table G 48"))
    df = pd.DataFrame(out)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {df['geo_id'].nunique()} villages, {df['count'].sum():,} people")
    print(df.groupby("source_category")["count"].sum())


if __name__ == "__main__":
    main()
