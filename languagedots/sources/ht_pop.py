"""Haiti: everyone on Haitian Creole, per department, on the COD-PS 2024 department populations
(IHSI's projections as published on HDX) -> data/normalized/ht.csv.

    python sources/ht_pop.py

NO LANGUAGE QUESTION. The 2003 RGPH asked literacy only; the fifth census was never held. Haitian
Creole is the first language of practically every Haitian; French, the other official language,
is learned at school (AGENT_BRIEF §2, learned second languages are not home languages). So the
only number needed is population. religiondots' do_lookup-style `ht_lookup.csv` carries the same
COD-PS figures as `pop_2024`; this script reads the HDX file itself (religiondots' download,
read-only) and asserts the two agree.

CHECKS: 10 departments; the file's total equals the lookup's per department.
"""
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD, RD_GEO  # noqa: E402

RAW = RD / "data" / "raw" / "ht" / "hti_admpop_adm1_2024.csv"
OUT = HERE / "data" / "normalized" / "ht.csv"


def main():
    p = pd.read_csv(RAW, dtype={"ADM1_PCODE": str})
    assert len(p) == 10, len(p)
    lut = pd.read_csv(RD_GEO / "ht" / "ht_lookup.csv", dtype=str)
    lut["pop_2024"] = lut["pop_2024"].astype(int)
    m = lut.merge(p[["ADM1_PCODE", "T_TL"]], left_on="geo_id", right_on="ADM1_PCODE",
                  how="outer", validate="1:1")
    assert m["T_TL"].notna().all() and m["name"].notna().all(), m
    assert (m["T_TL"] == m["pop_2024"]).all(), m[m["T_TL"] != m["pop_2024"]]
    df = pd.DataFrame(dict(geo_id=m["geo_id"], geo_level="departement", geo_name=m["name"],
                           source_category="Kreyòl ayisyen", count=m["T_TL"].astype(int),
                           tier="derived", source_id="cod_ps_hti_2024", year=2024,
                           note="no language question; everyone drawn as Haitian Creole"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(df)} departments, {df['count'].sum():,} people")


if __name__ == "__main__":
    main()
