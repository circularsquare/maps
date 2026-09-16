"""China: the share of people naming no religion who keep a religious home altar, by province, from
CGSS 2010's ISSP religion module. Spec §3.13, Anita 2026-09-15.

Writes `data/normalized/cn_altar.csv`, a table of COEFFICIENTS like `cn_cgss.csv`: province ->
the share of the `unknown` residual that `countries/cn.py` moves to `chinesefolk`. The residual is
what is left after the CGSS naming layer is carved, so it stands for people who named no religion.

THE ITEM. `na`, 您家里是否有出于信仰宗教的原因而设的神龛、祭坛、或者摆放宗教物品 ("for religious reasons,
is there a shrine, an altar or a religious object in your home"), put to the ISSP 2008 module's
subsample, 4,203 of 11,783. Eligible here: the module's respondents whose `a5` is 不信仰宗教 and whose
`a4` (您的民族是) is 汉.

HAN ONLY, 2026-09-15 (session cb8b206e-folkfix). `countries/cn.py` applies the share to the Han row's
residual alone, so the rate is the Han rate. The minority respondents answer the item differently: 240
of the 3,622 no-religion respondents, 16.8% with an altar weighted against the Han 13.7%, and in Guangxi
63 of them with 17 altars pull the province from a Han 6.1% to 17.2%. sources/cn.md §11.

THE TEST. `folk_altar.rate_test` over provinces, with the county (`s42`, CGSS 2010's primary sampling
unit) as the resampling unit. A community code (`s44`) would give more halves and a test that
passes too easily, because communities in one county were drawn together.

WHAT IS NOT HERE.
  * Xizang, dropped for `cn_cgss.DROP_PROVINCES`'s reason: §14.5 draws Tibet from ethnicity, and a
    CGSS layer there would restate it. `countries/cn.py` reads a missing province as zero.
  * Any other wave. CGSS 2018 carries the item (C21) with examples in the wording, and is not open.

Usage:
    python sources/cn_altar.py      rebuild data/normalized/cn_altar.csv
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import folk_altar  # noqa: E402
from cn_cgss import CN2EN, DROP_PROVINCES, _single  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw", "cn", "cgss", "cgss2010.dta")
OUT = os.path.join(ROOT, "data", "normalized", "cn_altar.csv")
SOURCE_ID = "cn_cgss_2010_issp_altar"

# Pinned after the first run (2026-09-15: median +0.494 against a null 95th of +0.227, p 0.0025,
# chi-square 1e-35, largest county 3%): a change is a change in what China's layer claims to know.
VERDICT = "own geography"
# The national rate, weighted, among eligible respondents. Measured 13.9% before this file existed
# (sources/folk_practice.md section 3); a load that drifts far from it has read the wrong column.
NATIONAL_RANGE = (0.10, 0.18)
# `a4`'s label for Han. 10,662 of 11,783 respondents; a load that finds far fewer has the wrong label.
HAN = "汉"


def load():
    if not os.path.exists(RAW):
        raise SystemExit(f"missing {RAW}; see sources/cn_cgss.md for the fetch")
    cat = pd.read_stata(RAW, columns=["s41", "a5", "a4"], convert_categoricals=True)
    num = pd.read_stata(RAW, columns=["na", "s42", "WEIGHT"], convert_categoricals=False)
    d = pd.DataFrame({
        "prov": cat["s41"].astype(str).map(CN2EN),
        "rel": cat["a5"].map(_single),
        "han": cat["a4"].astype(str) == HAN,
        "na": pd.to_numeric(num["na"], errors="coerce"),
        "psu": pd.to_numeric(num["s42"], errors="coerce"),
        "w": pd.to_numeric(num["WEIGHT"], errors="coerce"),
    })
    if d["prov"].isna().any():
        raise SystemExit(f"unmapped CGSS 2010 province labels: {sorted(set(cat['s41'][d['prov'].isna()].astype(str)))}")
    if d["w"].isna().any() or (d["w"] <= 0).any():
        raise SystemExit("CGSS 2010 WEIGHT missing or non-positive")
    return d


def main():
    d = load()
    module = d["na"].isin([1, 2])
    print(f"CGSS 2010: {len(d):,} respondents, {int(module.sum()):,} in the ISSP module")
    m = d[module]
    W = m["w"].sum()
    print(f"  module, weighted: home altar {100 * (m['w'] * (m['na'] == 1)).sum() / W:.1f}%, "
          f"no religion {100 * (m['w'] * (m['rel'] == 'none')).sum() / W:.1f}%")

    if int(d["han"].sum()) < 10000:
        raise SystemExit(f"only {int(d['han'].sum()):,} respondents read as Han ({HAN!r}); check a4's labels")
    nonhan = m[(m["rel"] == "none") & ~m["prov"].isin(DROP_PROVINCES) & ~m["han"]]
    print(f"  no-religion respondents in the module who are not Han, left out: {len(nonhan):,}")
    e = m[(m["rel"] == "none") & ~m["prov"].isin(DROP_PROVINCES) & m["han"]].copy()
    if e["psu"].isna().any():
        raise SystemExit("eligible respondents with no county code (s42)")
    units = sorted(e["prov"].unique())
    if len(units) != 30:
        raise SystemExit(f"{len(units)} provinces have eligible respondents, expected the 30 besides Xizang")
    uidx = e["prov"].map({p: i for i, p in enumerate(units)}).to_numpy()
    hit = (e["na"] == 1).to_numpy()
    cluster = (e["prov"] + "|" + e["psu"].astype(int).astype(str)).to_numpy()

    t = folk_altar.rate_test(hit, uidx, cluster, len(units))
    print(f"\n  eligible (no religion, Han, in the module, Xizang dropped): {t['n']:,}, of whom {t['k']:,} "
          f"keep an altar; {t['clusters']} counties")
    print(f"  split-half over {folk_altar.N_SPLITS} random halves of the counties, regrouping null "
          f"{folk_altar.N_NULL} draws: median Spearman {t['median']:+.3f}, null 95th {t['q95']:+.3f}, "
          f"p {t['p']:.4f}; chi-square p {t['chi_p']:.2e}; largest county {t['cell']:.0%} of altar "
          f"holders; top province {units[t['top']]}, largest county there {t['top_cell']:.0%}")
    print(f"  -> {t['verdict']}")
    if VERDICT is not None and t["verdict"] != VERDICT:
        raise SystemExit(f"the test now says {t['verdict']!r}, pinned {VERDICT!r}. That changes what "
                         "China's folk layer claims: read the table, then update VERDICT, this "
                         "docstring and sources/folk_practice.md deliberately.")

    k = np.bincount(uidx, weights=hit.astype(float), minlength=len(units))
    n = np.bincount(uidx, minlength=len(units)).astype(float)
    w = e["w"].to_numpy()
    kw = np.bincount(uidx, weights=w * hit, minlength=len(units))
    nw = np.bincount(uidx, weights=w, minlength=len(units))
    rate_w = kw / nw
    national = float((w * hit).sum() / w.sum())
    lo, hi = NATIONAL_RANGE
    if not lo <= national <= hi:
        raise SystemExit(f"national rate {national:.1%} is outside {lo:.0%}-{hi:.0%}; check the columns")
    psus = e.groupby("prov")["psu"].nunique().reindex(units).to_numpy()

    if t["verdict"] == "own geography":
        share, M = folk_altar.shrink(k, n, rate_w, national)
        method = f"province rate shrunk toward national, prior strength {M:.1f} respondents"
    else:
        share, M = np.full(len(units), national), None
        method = f"national rate ({t['verdict']})"
    print(f"  national weighted rate {national:.1%}; {method}")

    rows = []
    for i, p in enumerate(units):
        rows.append(dict(
            province=p, share=round(float(share[i]), 6), raw_share=round(float(rate_w[i]), 6),
            n=int(n[i]), k=int(k[i]), psus=int(psus[i]), method=method, basis="practice",
            source_id=SOURCE_ID,
            note=f"CGSS 2010 ISSP module, na; {int(k[i])} of {int(n[i])} Han no-religion respondents "
                 f"keep a religious altar, {int(psus[i])} counties"))
    out = pd.DataFrame(rows).sort_values("share", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} provinces)")
    print(f"    {'province':<16}{'n':>5}{'k':>5}{'counties':>9}{'raw':>8}{'drawn':>8}")
    for r in out.itertuples():
        print(f"    {r.province:<16}{r.n:>5}{r.k:>5}{r.psus:>9}{r.raw_share * 100:7.1f}%{r.share * 100:7.1f}%")


if __name__ == "__main__":
    main()
