"""Bahrain: the 2020 census tables religiondots already parsed and checked, copied out as a plain
CSV for sources/bh_build.py. Run as its own process (bh_build.py does that): it loads
religiondots' sources/bh.py, which puts religiondots' taxonomy folder on sys.path.

READ-ONLY on religiondots: its parser and checks are called, nothing is written there.

Writes data/raw/bh/bh_groups.csv: 4 governorates x nationality group (Bahraini, Gulf
Co-operative Countries, Other Arabs, Asian, African, European, North American, Others) x sex,
census 2020 (data.gov.bh), checked by religiondots against the governorate and religion tables.
"""
import importlib.util
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RD_BH = ROOT.parent / "religiondots" / "sources" / "bh.py"
OUT = ROOT / "data" / "raw" / "bh"


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    spec = importlib.util.spec_from_file_location("rd_bh", RD_BH)
    bh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bh)
    _rel, _gov, grp = bh.load_census()
    df = grp.rename("people").reset_index()
    df.columns = ["governorate", "group", "sex", "people"]
    if int(df["people"].sum()) != bh.CENSUS_TOTAL:
        raise SystemExit("groups do not sum to the census")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "bh_groups.csv", index=False)
    print(f"wrote {OUT / 'bh_groups.csv'} ({len(df)} cells, {bh.CENSUS_TOTAL:,} people)")


if __name__ == "__main__":
    main()
