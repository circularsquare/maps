"""
Egypt's Christians split into the Coptic Orthodox Church and the rest, from the Global
Flourishing Study wave 1 (2023). Writes data/normalized/eg_churches.csv, one national share that
countries/eg.py applies to every governorate's drawn Christians. Session `fafd1067-copts`,
2026-10-04, on Anita's "would be really nice if we could place copts vs noncopts in egypt".
`sources/eg.md`, "The Copts" has the record.

    python sources/eg_churches.py            (needs data/raw/gfs/gfs_all_countries_wave2.csv:
                                               `python sources/jp_gfs.py --fetch`, CC BY, OSF vrejf)

THE ITEM. `REL3_Y1` (GFS codebook wave 2, OSF 285w7, p.21), asked of everyone who gave
Christianity in `REL2_Y1`: "Which of the following denominations or churches do you most identify
with, if any?". Egypt is `COUNTRY` 4, 4,729 respondents, 131 of them Christian (2.54% weighted).

    code   answer                               n    weighted share of Christians
    2      Orthodox                           109    86.49%
    1      Catholic                             4     2.76%
    3      Anglican/Episcopal                   3     1.70%
    4      Presbyterian/Reformed                1     0.69%
    9      Independent/Holiness/Evangelical     2     0.50%
    97     no denomination in particular       10     7.36%
    99     refused                              2     0.51%

`Orthodox` goes to `christianity.oriental.coptic` (taxonomy/eg2022.py REVIEW has why: in Egypt the
answer names the Coptic Orthodox Church for all but a few thousand Greek and Armenian Orthodox).
Everything else, named or not, stays on `christianity`: four Catholics and six Protestants cannot
carry nodes of their own, and the brief leaves the no-church share on the parent.

NATIONAL, NOT BY GOVERNORATE. 131 Christians over 18 GFS regions, 42 of 125 PSUs; random halves
within region over the 11 regions with 4 or more Christians give a median Spearman of +0.515 on the
Orthodox share, below the bar for 11 units (+0.536, `spearman_null.critical_rho(11)`) even before
allowing that the halves share PSUs and so flatter. So one share for the whole country.

THE WITNESSES, printed and not drawn (`witnesses()`):
  * Arab Barometer wave V (2018-19), `Q1012A`, 268 Egyptian Christians: Orthodox 71.7% weighted,
    Catholic 16.2%, Armenian 0.6%, don't know or refused 11.5%. Its Catholic share is three times
    the Catholic Church's own count (348,000 Catholics in Egypt in 2023, all rites and foreigners,
    GCatholic from the Annuarium, about 5% of the Christians drawn here), and a church's own count
    is the high side; so wave V's card is not read for the level (sources/eg.md).
  * Arab Barometer wave III (2013), `q1012a`: 25 of 69 Christians answered, Orthodox 20, Catholic 4.
  * Wave VII's `Q1012A_CHRISTIAN` is empty for all 66 Egyptian Christians; WVS 7 Egypt (2018) codes
    its 37 Christians `Other Christian; nfd` (IHSN catalogue 11567, Q289CS9).
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GFS_CSV = os.path.join(ROOT, "data", "raw", "gfs", "gfs_all_countries_wave2.csv")
AB = os.path.join(ROOT, "data", "raw", "arabbarometer")
OUT = os.path.join(ROOT, "data", "normalized", "eg_churches.csv")

GFS_COUNTRY = 4
GFS_N = 4_729
GFS_CHRISTIANS = 131
ORTHODOX = 2
COPTIC = "Christian: Orthodox (GFS 2023)"
REST = "Christian: other church or none named (GFS 2023)"
# Measured 2026-10-04: 0.8649. The build stops if the file on disk gives something else.
EXPECT = 0.8649


def gfs():
    g = pd.read_csv(GFS_CSV, usecols=["COUNTRY", "REL2_Y1", "REL3_Y1", "ANNUAL_WEIGHT_C1"],
                    low_memory=False)
    for c in g.columns:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    e = g[g["COUNTRY"] == GFS_COUNTRY]
    if len(e) != GFS_N:
        raise SystemExit(f"GFS Egypt has {len(e):,} respondents, this was written against {GFS_N:,}")
    c = e[e["REL2_Y1"] == 1]
    if len(c) != GFS_CHRISTIANS or c["REL3_Y1"].isna().any():
        raise SystemExit(f"GFS Egypt has {len(c)} Christians (expected {GFS_CHRISTIANS}) or a "
                         "Christian with no REL3 answer")
    w = c["ANNUAL_WEIGHT_C1"]
    share = float(w[c["REL3_Y1"] == ORTHODOX].sum() / w.sum())
    neff = float(w.sum() ** 2 / (w ** 2).sum())
    print(f"  GFS 2023 Egypt: {len(e):,} respondents, {len(c)} Christian "
          f"({100 * e.loc[e['REL2_Y1'] == 1, 'ANNUAL_WEIGHT_C1'].sum() / e['ANNUAL_WEIGHT_C1'].sum():.2f}% weighted)")
    t = c.groupby("REL3_Y1")["ANNUAL_WEIGHT_C1"].agg(["size", "sum"])
    for code, r in t.iterrows():
        print(f"    REL3 {int(code):>3}  n={int(r['size']):>3}  {100 * r['sum'] / w.sum():6.2f}%")
    half = 1.96 * np.sqrt(share * (1 - share) / neff)
    print(f"  Orthodox {100 * share:.2f}% of Christians, Kish n_eff {neff:.0f}, "
          f"95% about +-{100 * half:.1f} points")
    if abs(share - EXPECT) > 5e-5:
        raise SystemExit(f"the Orthodox share is {share:.4f}, written against {EXPECT}")
    return share, len(c), neff


def witnesses():
    """Arab Barometer waves III and V, Christians' denomination answer. Printed, not used."""
    try:
        import pyreadstat
    except ImportError:
        print("  (witnesses skipped: no pyreadstat)")
        return
    for wave, f, rel, den in [("III", "ABIII_English.sav", "q1012", "q1012a"),
                              ("V", "ArabBarometer_WaveV_English_v2.sav", "Q1012", "Q1012A")]:
        p = os.path.join(AB, f)
        if not os.path.exists(p):
            print(f"  (wave {wave} witness skipped: {f} absent; python sources/eg.py --fetch)")
            continue
        df, _ = pyreadstat.read_sav(p, usecols=["country", rel, den, "wt"], apply_value_formats=True)
        ch = df[df["country"].astype(str).str.contains("Egypt") &
                df[rel].astype(str).str.contains("hristian")].copy()
        ch["d"] = ch[den].astype(str)
        t = ch.groupby("d")["wt"].agg(["size", "sum"])
        print(f"  Arab Barometer wave {wave}, {len(ch)} Christians: " + ", ".join(
            f"{k} {int(r['size'])} ({100 * r['sum'] / ch['wt'].sum():.1f}%)" for k, r in t.iterrows()))


def main():
    if not os.path.exists(GFS_CSV):
        raise SystemExit("needs data/raw/gfs/gfs_all_countries_wave2.csv; python sources/jp_gfs.py --fetch")
    share, n, neff = gfs()
    witnesses()
    pd.DataFrame([
        {"source_category": COPTIC, "share_of_christians": round(share, 6), "n_christians": n,
         "n_eff": round(neff, 1), "source_id": "gfs_wave1_2023_rel3"},
        {"source_category": REST, "share_of_christians": round(1 - share, 6), "n_christians": n,
         "n_eff": round(neff, 1), "source_id": "gfs_wave1_2023_rel3"},
    ]).to_csv(OUT, index=False)
    print(f"  -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
