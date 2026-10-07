"""British Virgin Islands: first language from the 2010 census's country of birth by island
-> data/normalized/vg.csv.

    python sources/vg_census.py

NO CENSUS LANGUAGE QUESTION (2010). Built as Barbados and St Kitts (sources/bb.md, kn.md): the
native-born on Virgin Islands Creole (Glottolog virg1240, which covers both the US and British
Virgin Islands; node from tree.d/ag.txt), the foreign-born on their birth country's language.
Every row `derived`.

THE TABLE: Virgin Islands 2010 Population and Housing Census Report, Table 84 "Grouped Country of
Birth by Island" (report p. 65, PDF p. 72), religiondots' copy of
unstats.un.org/unsd/demographic/sources/census/wphc/BVI/VGB-2016-09-08.pdf (read only).
22 grouped birthplaces x 7 islands (Anegada, Cooper Island, Great Camanoe Island, Jost Van Dyke,
Tortola, Virgin Gorda, Yachts). Cooper Island, Great Camanoe and Yachts (50 people) are drawn with
Tortola, as religiondots does (they have no Kontur hex).

CHECKS: every row's islands sum to its printed total; the island columns sum to the printed
Total row; the grand total is 28,054; Table 83's national figures agree with Table 84's totals.
"""
import re
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

RAW = RD / "data" / "raw" / "terr" / "VGB-2016-09-08.pdf"
OUT = HERE / "data" / "normalized" / "vg.csv"
TOTAL = 28_054
ISL = ["Anegada", "Cooper Island", "Great Camanoe Island", "Jost Van Dyke", "Tortola",
       "Virgin Gorda", "Yachts"]
UNIT = {"Anegada": "VG-ANEGADA", "Jost Van Dyke": "VG-JOST-VAN-DYKE", "Tortola": "VG-TORTOLA",
        "Virgin Gorda": "VG-VIRGIN-GORDA", "Cooper Island": "VG-TORTOLA",
        "Great Camanoe Island": "VG-TORTOLA", "Yachts": "VG-TORTOLA"}
LABELS = ["Virgin Islands", "Other Caribbean", "Dominica", "Grenada", "Jamaica",
          "St Kitts and Nevis", "St Lucia", "St Vincent and Grenadines", "Trinidad and Tobago",
          "Dominican Republic", "Guyana", "United States of America",
          "United States Virgin Islands", "Puerto Rico", "United Kingdom", "Europe",
          "Latin America", "Asia", "Africa", "Pacific", "Middle East", "Overseas Territories",
          "Other Countries", "Not Stated", "Total"]


def table84():
    import fitz
    t = fitz.open(RAW)[71].get_text()
    toks = [x.strip() for x in t.split("\n") if x.strip()]
    i = toks.index("Yachts") + 1
    rows = {}
    for lab in LABELS:
        name = " ".join(toks[i:i + 3])
        # the label may span tokens ("United  Kingdom'"); consume until a number
        j = i
        while not re.fullmatch(r"[\d,]+", toks[j]):
            j += 1
        got = re.sub(r"\s+", " ", " ".join(toks[i:j])).strip(" '")
        assert got == lab, (got, lab, name)
        nums = toks[j:j + 15]
        cnt = [int(nums[k].replace(",", "")) for k in range(0, 14, 2)]
        tot = int(nums[14].replace(",", ""))
        assert sum(cnt) == tot, (lab, cnt, tot)
        rows[lab] = dict(zip(ISL, cnt))
        i = j + 15
    return rows


def main():
    rows = table84()
    tot = rows.pop("Total")
    for k in ISL:
        assert sum(r[k] for r in rows.values()) == tot[k], k
    assert sum(tot.values()) == TOTAL
    t83 = re.sub(r"\s+", " ", __import__("fitz").open(RAW)[70].get_text())
    for lab in ("Guyana", "Jamaica", "Dominican Republic", "Puerto Rico"):
        n = sum(rows[lab].values())
        assert f"{lab} | {n:,}".replace(" | ", " ") in t83, lab
    out = []
    for lab, r in rows.items():
        if lab == "Not Stated":
            continue
        for isl, n in r.items():
            if n:
                out.append(dict(geo_id=UNIT[isl], geo_name=isl, source_category=lab, count=n))
    df = pd.DataFrame(out).groupby(["geo_id", "source_category"], as_index=False)["count"].sum()
    df["geo_level"] = "island"
    df["tier"] = "derived"
    df["year"] = 2010
    df.to_csv(OUT, index=False, encoding="utf-8")
    ns = sum(rows["Not Stated"].values())
    print(f"wrote {OUT}: {len(df)} rows, {df['count'].sum():,} people ({ns} not stated, not drawn)")
    print(df.groupby("geo_id")["count"].sum().to_string())


if __name__ == "__main__":
    main()
