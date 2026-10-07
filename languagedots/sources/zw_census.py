"""Zimbabwe: 2022 PHC mother tongue by province (ZIMSTAT main report, Table 2.17).

    python sources/zw_census.py      -> data/normalized/zw.csv  (province x mother tongue)

The report is religiondots' data/raw/zw/zw_phc2022_report.pdf (read-only; ZIMSTAT,
https://www.zimstat.co.zw/wp-content/uploads/Census/2022_PHC_Report_27012023_Final.pdf),
p. 148 of the PDF (printed 123). "Mother tongue is the language usually spoken in the
individual's home in his/her early childhood" (p. 41). 17 categories x 10 provinces; the
table's total is 13,913,253 of 15,178,957 enumerated, which matches people aged 3 and over
(the education module's universe; 13,102,643 are 5+, p. 126), so the table is that universe.

CHECKS: every row sums to its Total column, every column to the Total row; the report's own
text (Shona 80.9%, Ndebele 11.5%); every province's table total is 0.90-0.95 of its census
population (religiondots' normalized zw.csv, read-only); the Youth thematic report's
province percentages (a sub-universe, so a loose check) put Ndau and Tonga in the same
provinces.
"""
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RD = HERE.parent / "religiondots"
PDF = RD / "data" / "raw" / "zw" / "zw_phc2022_report.pdf"
RD_NORM = RD / "data" / "normalized" / "zw.csv"
OUT = HERE / "data" / "normalized" / "zw.csv"
PAGE = 147                      # 0-based: printed page 123
TOTAL = 13_913_253
CENSUS = 15_178_957

# Table 2.17's column order -> religiondots' province geo_id (its zw_lookup.csv)
PROVINCES = [("ZW01", "Bulawayo"), ("ZW02", "Manicaland"), ("ZW03", "Mashonaland Central"),
             ("ZW04", "Mashonaland East"), ("ZW05", "Mashonaland West"),
             ("ZW06", "Matabeleland North"), ("ZW07", "Matabeleland South"),
             ("ZW08", "Midlands"), ("ZW09", "Masvingo"), ("ZW10", "Harare")]
LANGS = ["Shona", "Ndebele", "English", "Kalanga", "Koisan", "Nambya", "Ndau", "Chibarwe",
         "Shangani", "Chewa", "Sign Language", "Sotho", "Tonga", "Tswana", "Venda", "Xhosa",
         "Other"]


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def parse():
    import fitz
    d = fitz.open(str(PDF))
    t = d[PAGE].get_text()
    say("Table 2.17: Distribution of Population by Mother Tongue and Province" in t,
        "p. 148 is Table 2.17")
    lines = [" ".join(x.split()) for x in t.splitlines()]
    lines = [x for x in lines if x]
    rows = {}
    for name in LANGS + ["Total"]:
        i = lines.index(name, lines.index("Shona") if name != "Shona" else 0)
        nums = lines[i + 1:i + 12]
        say(all(re.fullmatch(r"[\d,]+", n) for n in nums), f"{name}: 11 figures follow")
        rows[name] = [int(n.replace(",", "")) for n in nums]
    return rows


def main():
    rows = parse()
    tot = rows.pop("Total")
    for k, v in rows.items():
        say(sum(v[:10]) == v[10], f"{k}: provinces sum to the Total column ({v[10]:,})")
    for j in range(11):
        s = sum(v[j] for v in rows.values())
        say(s == tot[j], f"column {j}: languages sum to the Total row ({tot[j]:,})")
    say(tot[10] == TOTAL, f"table total {tot[10]:,}")
    sh, nd = rows["Shona"][10] / TOTAL, rows["Ndebele"][10] / TOTAL
    say(round(sh * 100, 1) == 80.9 and round(nd * 100, 1) == 11.5,
        f"report text: Shona {sh:.2%} (80.9), Ndebele {nd:.2%} (11.5)")

    rd = pd.read_csv(RD_NORM, keep_default_na=False, na_values=[""])
    pop = rd[rd["source_category"] == "Total"].set_index("geo_id")["count"]
    say(int(pop.sum()) == CENSUS, f"religiondots' province totals sum to {CENSUS:,}")
    for j, (g, n) in enumerate(PROVINCES):
        r = tot[j] / pop[g]
        say(0.90 <= r <= 0.95, f"{n}: table {tot[j]:,} / census {int(pop[g]):,} = {r:.3f}")

    out = []
    for k, v in rows.items():
        for j, (g, n) in enumerate(PROVINCES):
            if v[j] > 0:
                out.append((g, "province", n, k, v[j], "measured", "zw_phc2022_t2_17", 2022))
    df = pd.DataFrame(out, columns=["geo_id", "geo_level", "geo_name", "source_category",
                                    "count", "tier", "source_id", "year"])
    say(int(df["count"].sum()) == TOTAL, f"normalized total {int(df['count'].sum()):,}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows)")
    nat = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    for k, v in nat.items():
        print(f"    {k:14s} {v:>11,}  {v / TOTAL:6.2%}")


if __name__ == "__main__":
    main()
