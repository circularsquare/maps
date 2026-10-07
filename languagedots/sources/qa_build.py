"""Qatar, 2020 census: Qataris and non-Qataris by municipality and sex, read as languages.
-> data/normalized/qa.csv

    python sources/qa_build.py

Nobody in Qatar is asked a language, and Qatar publishes no count of its citizens as such. Built by
the Gulf home-mix method (sources/gulf_mix.py, sources/gulf.md), every row `derived`. From the
Planning and Statistics Authority's *Census 2020 Detailed Results*
(data/raw/qa/Census_Final_Results.xlsx, npc.qa):

  * Table 1: population by municipality and sex (2,846,118);
  * Table 32: Qatari population aged 10+ by municipality and sex (246,256);
  * Qataris under 10, which no table gives: each municipality's under-10s (Tables 4 and 5, by
    sex) times the Qatari share of 10-14 year olds nationally (Table 20's 37,770 Qataris of Table
    3's 128,748). Children are where Qataris are most of the population, and that share is the
    nearest one the census prints;
  * non-Qataris = Table 1 less Qataris, by sex; each sex at UN DESA's 2020 mix of Qatar's
    migrants of that sex (no nationality is published at all), each origin on its language or
    home mix, DESA's unnamed `Others` on `other`.
Qataris on Gulf Arabic.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gulf_mix as g  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = HERE.parent
XLSX = ROOT / "data" / "raw" / "qa" / "Census_Final_Results.xlsx"
OUT = ROOT / "data" / "normalized" / "qa.csv"
TOTAL = 2_846_118
QATARI_10P = 246_256
MUNIS = ["Doha", "Al Rayyan", "Al Wakra", "Umm Slal", "Al Khor and Al Thakhira", "Al Shamal",
         "Al Daayen", "Al Sheehaniya"]
SHORT = {"Al Khor": "Al Khor and Al Thakhira"}    # the age tables' header spells it short


def sheet(n):
    return pd.read_excel(XLSX, sheet_name=str(n), header=None)


def muni_cols(t):
    """{municipality: column} from the header cell holding each English name."""
    cols = {}
    for i in range(4, 8):
        for j, v in enumerate(t.iloc[i]):
            s = str(v)
            for m in MUNIS + list(SHORT):
                if s.split("\n")[-1].strip().startswith(m) and "\n" in s:
                    cols.setdefault(SHORT.get(m, m), j)
    # "Al Khor ..." also starts with "Al Khor"; "Al Rayyan" etc. are unique
    if len(cols) != 8:
        raise SystemExit(f"municipality columns not found: {cols}")
    return cols


def table1():
    out = {}
    for _i, r in sheet(1).iterrows():
        lab = str(r[0]).strip()
        if lab in MUNIS:
            out[lab] = dict(total=int(r[3]), men=int(r[2]), women=int(r[1]))
    if sum(v["total"] for v in out.values()) != TOTAL:
        raise SystemExit("Table 1 does not sum to 2,846,118")
    return out


def qatari_10p():
    t = sheet(32)
    cols = muni_cols(t)
    out = {}
    for i in range(len(t)):
        if str(t.iat[i, 0]).strip() == "Total" and str(t.iat[i, 1]).strip() == "Total":
            for k, sex in enumerate(("total", "men", "women")):
                for m, j in cols.items():
                    out.setdefault(m, {})[sex] = int(t.iat[i + k, j])
            break
    if sum(v["total"] for v in out.values()) != QATARI_10P or \
            any(v["men"] + v["women"] != v["total"] for v in out.values()):
        raise SystemExit("Table 32's totals are not as pinned")
    return out


def under10(n):
    """Table 4 (men) or 5 (women): {municipality: people under 10}."""
    t = sheet(n)
    cols = muni_cols(t)
    out = dict.fromkeys(cols, 0)
    rows = 0
    for i in range(len(t)):
        if str(t.iat[i, 0]).strip() in ("Under 1", "1 - 4", "5 - 9"):
            rows += 1
            for m, j in cols.items():
                out[m] += int(t.iat[i, j])
    if rows != 3:
        raise SystemExit(f"Table {n}: {rows} under-10 rows")
    return out


def share_10_14():
    t3 = sheet(3)
    tot = next(int(t3.iat[i, 9]) for i in range(len(t3)) if str(t3.iat[i, 0]).strip() == "10 - 14")
    t20 = sheet(20)
    hdr = next(j for j, v in enumerate(t20.iloc[5]) if str(v).strip() == "14 - 10")
    row = next(i for i in range(len(t20)) if str(t20.iat[i, 0]).strip() == "Both Sexes")
    q = int(t20.iat[row, hdr])
    print(f"  Qataris 10-14: {q:,} of {tot:,} ({q / tot:.1%}), Table 20 over Table 3")
    return q / tot


def main():
    t1 = table1()
    q10 = qatari_10p()
    u_m, u_f = under10(4), under10(5)
    s = share_10_14()
    units = pd.DataFrame(index=MUNIS)
    units["q_men"] = [q10[m]["men"] + s * u_m[m] for m in MUNIS]
    units["q_women"] = [q10[m]["women"] + s * u_f[m] for m in MUNIS]
    units["q"] = (units["q_men"] + units["q_women"]).round().astype(int)
    units["men"] = [t1[m]["men"] - units.at[m, "q_men"] for m in MUNIS]
    units["women"] = [t1[m]["women"] - units.at[m, "q_women"] for m in MUNIS]
    # whole people: the non-Qatari sexes absorb the rounding of the Qatari estimate
    units["women"] = [t1[m]["total"] - units.at[m, "q"] - units.at[m, "men"] for m in MUNIS]
    if (units[["men", "women"]] <= 0).any().any():
        raise SystemExit("a municipality has more Qataris of a sex than people")
    print(f"  Qataris estimated: {int(units['q'].sum()):,} ({units['q'].sum() / TOTAL:.1%}); "
          f"10+ counted {QATARI_10P:,}, under-10 at {s:.1%} of each municipality's under-10s")

    origins, others, _ = g.desa("Qatar", year=2020)
    nat_n = {iso: v[0] for iso, v in origins.items()}
    fixed = {iso: g.origin_mix(iso, "QA", n) for iso, n in nat_n.items()}
    mixes = {}
    for col, k in (("men", 1), ("women", 2)):
        w = {iso: v[k] for iso, v in origins.items()}
        w["node:" + g.OTHER] = others[k]
        mixes[col] = g.blend(w, "QA", fixed)
    fc = g.spread(units[["men", "women"]], mixes)

    rows = [dict(geo_id=m, geo_level="municipality", geo_name=m, origin="Qatari (estimated)",
                 source_category=g.GULF_ARABIC, count=int(units.at[m, "q"])) for m in MUNIS]
    for gid, row in fc.iterrows():
        for n, c in row.items():
            if c > 0:
                rows.append(dict(geo_id=gid, geo_level="municipality", geo_name=gid,
                                 origin="non-Qatari", source_category=n, count=int(c)))
    out = pd.DataFrame(rows)
    out["tier"] = "derived"
    out["year"] = 2020
    if int(out["count"].sum()) != TOTAL:
        raise SystemExit("output does not sum to the census")
    for m in MUNIS:
        if int(out.loc[out["geo_id"] == m, "count"].sum()) != t1[m]["total"]:
            raise SystemExit(f"{m} does not sum to Table 1")
    out.to_csv(OUT, index=False, encoding="utf-8")
    g.report(out, TOTAL, f"wrote {OUT}")
    for col in ("men", "women"):
        top = sorted(mixes[col].items(), key=lambda kv: -kv[1])[:6]
        print(f"    non-Qatari {col}: " + ", ".join(f"{n.split('.')[-1]} {x:.1%}" for n, x in top))


if __name__ == "__main__":
    main()
