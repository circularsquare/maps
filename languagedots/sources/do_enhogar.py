"""Dominican Republic: ONE's ENHOGAR-MICS6 2019 household survey, mother tongue of the household
head (HC1B), per province, on the 2022 census province populations -> data/normalized/do.csv.

    python sources/do_enhogar.py

NO CENSUS LANGUAGE QUESTION. Neither the 2010 nor the 2022 census asks language (queue note,
coverage/scout_2026-10-05.md). ENHOGAR-MICS6 2019 does: household questionnaire item HC1B,
"Cual es el idioma materno del jefe o la jefa del hogar, es decir, el que hablaba cuando era
nino(a)?", answers ESPANOL / CREOL / INGLES / FRANCES / OTRO IDIOMA. 31,488 completed households,
all 32 provinces (HH7A). The microdata are religiondots' download (read-only):
../religiondots/data/raw/do/mics6_2019_hogares.csv (+ .sav for the labels).

THE MODEL. Each household is weighted by hhweight x HH48 (members), so a province's share is the
share of its people living in a household whose head's mother tongue is that language. The
shares are applied to the province's 2022 census population (religiondots' do_lookup.csv,
pop_2022, from ONE's X CNPV). Every row `modelled`.

CHECKS: 31,488 completed households; every one with an HC1B answer; 32 provinces, each with
more than 400 households; the province counts sum to the census total; the national Creole
share lands between 5% and 8% (ENI-2017 counted 7.5% of the population as Haitian-born or of
Haitian descent, which is the ceiling a head-language share should sit under or near).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD, RD_GEO  # noqa: E402

RAW = RD / "data" / "raw" / "do" / "mics6_2019_hogares.csv"
OUT = HERE / "data" / "normalized" / "do.csv"
LABELS = {"1": "Español", "2": "Creol", "3": "Inglés", "4": "Francés", "6": "Otro idioma"}


def main():
    h = pd.read_csv(RAW, dtype=str, low_memory=False)
    h = h[h["HH46"].str.strip() == "1"].copy()
    assert len(h) == 31_488, len(h)
    h["lang"] = h["HC1B"].str.strip().map(LABELS)
    assert h["lang"].notna().all(), h.loc[h["lang"].isna(), "HC1B"].value_counts()
    h["prov"] = h["HH7A"].str.strip()
    h["w"] = pd.to_numeric(h["hhweight"]) * pd.to_numeric(h["HH48"])
    n = h.groupby("prov").size()
    assert len(n) == 32 and n.min() > 400, n.describe()

    lut = pd.read_csv(RD_GEO / "do" / "do_lookup.csv", dtype=str)
    lut["pop_2022"] = lut["pop_2022"].astype(int)
    assert lut["enhogar_hh7a"].nunique() == 32
    share = h.pivot_table(index="prov", columns="lang", values="w", aggfunc="sum", fill_value=0)
    share = share.div(share.sum(axis=1), axis=0)

    rows = []
    for _, r in lut.iterrows():
        s = share.loc[r["enhogar_hh7a"]]
        hh = n[r["enhogar_hh7a"]]
        for lang, v in s.items():
            if v <= 0:
                continue
            rows.append(dict(geo_id=r["geo_id"], geo_level="provincia", geo_name=r["name"],
                             source_category=lang, count=round(v * r["pop_2022"]),
                             tier="modelled", source_id="enhogar_mics6_2019_hc1b", year=2019,
                             note=f"{v:.4f} of people in sampled households ({hh} households); "
                                  f"2022 census population {r['pop_2022']}"))
    df = pd.DataFrame(rows)
    total, census = df["count"].sum(), lut["pop_2022"].sum()
    assert abs(total - census) < 200, (total, census)
    nat = df.groupby("source_category")["count"].sum() / total
    print(nat.round(4).to_string())
    assert 0.05 < nat["Creol"] < 0.08, nat["Creol"]
    by = df.pivot_table(index="geo_name", columns="source_category", values="count",
                        aggfunc="sum", fill_value=0)
    print((by["Creol"] / by.sum(axis=1)).sort_values(ascending=False).round(3).head(8).to_string())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(df)} rows, {df['geo_id'].nunique()} provinces, {total:,} people")


if __name__ == "__main__":
    main()
