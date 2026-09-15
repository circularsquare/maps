"""Japan: does the 1996 prefecture pattern still hold? The Global Flourishing Study as a check. Draws nothing.

    python sources/jp_gfs.py --fetch     the GFS waves 1-2 public CSV, all countries, ~123 MB
    python sources/jp_gfs.py             the comparison

**THE SOURCE.** Global Flourishing Study (Gallup, Baylor, Harvard; Center for Open Science), wave 1,
Japan, 20,543 respondents, December 2022 to June 2023. Public on OSF since 2026-04-08 under
CC BY 4.0, no account (project c8hbk, component ge2x5, file `gfs_all_countries_wave2.csv`).
`REGION1` is the 47 prefectures in JIS order (901-947), `REL2` is *"What is your current
religion?"* with Buddhism and Shinto on the list, weight `ANNUAL_WEIGHT_C1`. An opt-in web panel,
weighted to region by age, sex, employment and marital status; not a probability sample.

**WHY IT IS A CHECK AND NOT AN INPUT** (sources/jp.md §8). Its list offers Buddhism by name, so it
reads Japan as 38% religious against JGSS's 25.6%, and its levels cannot be mixed in (spec §3.1a).
Its PATTERN could be, and was tested for exactly that:

  * the share naming any religion and the Buddhist share replicate between random halves of the
    sample across 47 prefectures (+0.69, +0.70 against a bar of 1.96/sqrt(46) = 0.29) and agree
    with NHK 1996 at +0.67 and +0.72, twenty-seven years apart. That is the evidence that the
    1996 pattern sources/jp_alloc.py spreads JGSS by is still broadly the pattern;
  * Christian (+0.03) and Shinto (+0.23) do not replicate, so it cannot place either;
  * at JGSS's six blocks it ranks with JGSS-2015 at +0.89, JGSS-2000/01 at +0.83 and NHK 1996 at
    +0.94, but it puts Hokkaido/Tohoku ABOVE the national share where all three probability
    samples put it below. Three unrelated samples agreeing against one opt-in panel is the case
    for keeping JGSS-2015 as the block level.
"""
import os
import sys
import urllib.request

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]], scipy below
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from jp_grid import PREFS, prefecture_population   # noqa: E402
import jp_alloc as ja                              # noqa: E402

CSV = os.path.join(ROOT, "data", "raw", "gfs", "gfs_all_countries_wave2.csv")
URL = "https://osf.io/download/vrejf/"
JAPAN = 9
NOT_ANSWERED = {98, 99, -98}
NONE, BUDDHISM, SHINTO, CHRISTIANITY = 97, 4, 9, 1
# JGSS-2000/01 believe or family religion by block, measured off Kimura (JGSS Research Series 2,
# 2003) figure 3 to about a point; the chart prints no values.
JGSS2000 = {"Hokkaido/Tohoku": 33.8, "Kanto": 23.0, "Chubu": 39.5, "Kinki": 40.2,
            "Chugoku/Shikoku": 42.0, "Kyushu": 41.8}


def fetch():
    if os.path.exists(CSV) and os.path.getsize(CSV) > 100_000_000:
        print(f"  have {CSV}")
        return
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(CSV + ".part", "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    os.replace(CSV + ".part", CSV)
    print(f"  got {CSV} ({os.path.getsize(CSV):,} bytes)")


def _sp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def main():
    from scipy.stats import chi2_contingency
    if "--fetch" in sys.argv:
        fetch()
    if not os.path.exists(CSV):
        raise SystemExit(f"missing {CSV}: run with --fetch")
    cols = ["ID", "COUNTRY", "REGION1_Y1", "REL2_Y1", "ANNUAL_WEIGHT_C1"]
    df = pd.read_csv(CSV, usecols=cols, low_memory=False)
    jp = df[pd.to_numeric(df["COUNTRY"], errors="coerce") == JAPAN].copy()
    if len(jp) != 20_543:
        raise SystemExit(f"GFS Japan has {len(jp):,} rows, this check was written against 20,543")
    jp["reg"] = pd.to_numeric(jp["REGION1_Y1"], errors="coerce")
    jp["rel"] = pd.to_numeric(jp["REL2_Y1"], errors="coerce")
    jp["w"] = pd.to_numeric(jp["ANNUAL_WEIGHT_C1"], errors="coerce")
    if not jp["reg"].between(901, 947).all():
        raise SystemExit("GFS Japan REGION1 is not 901-947 throughout")
    jp = jp[~jp["rel"].isin(NOT_ANSWERED) & jp["rel"].notna()].copy()
    jp["pref"] = [PREFS[int(c) - 901] for c in jp["reg"]]
    flags = {"religion": jp["rel"] != NONE, "buddhist": jp["rel"] == BUDDHISM,
             "shinto": jp["rel"] == SHINTO, "christian": jp["rel"] == CHRISTIANITY}
    for k, v in flags.items():
        jp[k] = v.astype(float)
    n = jp.groupby("pref").size().reindex(PREFS)
    wsh = pd.DataFrame({k: (jp[k] * jp["w"]).groupby(jp["pref"]).sum()
                        / jp["w"].groupby(jp["pref"]).sum() for k in flags}).reindex(PREFS)
    nat = {k: (jp[k] * jp["w"]).sum() / jp["w"].sum() for k in flags}
    print(f"GFS wave 1 Japan: {len(jp):,} answered; weighted " +
          ", ".join(f"{k} {100 * v:.1f}%" for k, v in nat.items()) +
          f"; per prefecture n {n.min()} to {n.max()}, median {n.median():.0f}")

    bar = 1.96 / np.sqrt(46)
    rng = np.random.default_rng(7)
    t = ja.nhk()
    nhk_col = {"religion": "total_measured", "buddhist": "buddhist_any", "shinto": "shinto",
               "christian": "christian"}
    print(f"\n47 prefectures (split-half bar {bar:.2f}):")
    for k in flags:
        yes = jp.groupby("pref")[k].sum().reindex(PREFS).to_numpy()
        chi2, p, dof, _ = chi2_contingency(np.vstack([yes, n.to_numpy() - yes]))
        rs = []
        for _ in range(200):
            m = rng.random(len(jp)) < 0.5
            rs.append(_sp(jp[m].groupby("pref")[k].mean().reindex(PREFS).fillna(0),
                          jp[~m].groupby("pref")[k].mean().reindex(PREFS).fillna(0)))
        print(f"    {k:<10} chi2 p {p:8.1e}  split-half {np.mean(rs):+.3f}  "
              f"{'replicates' if np.mean(rs) > bar else 'does NOT replicate'};  "
              f"vs NHK 1996 {_sp(wsh[k], t[nhk_col[k]]):+.3f}")

    pop = pd.Series(prefecture_population()).reindex(PREFS)
    rows = []
    for b, (s15, ps) in ja.BLOCKS.items():
        wp = pop[ps]
        rows.append(dict(block=b, gfs_2023=100 * (wsh.loc[ps, "religion"] * wp).sum() / wp.sum(),
                         jgss_2015=s15, jgss_2000=JGSS2000[b],
                         nhk_1996=(t.loc[ps, "total_measured"] * wp).sum() / wp.sum()))
    B = pd.DataFrame(rows).set_index("block")
    rel = B / B.mul(pd.Series({b: pop[ps].sum() for b, (_, ps) in ja.BLOCKS.items()}), axis=0) \
        .sum().div(pop.sum())
    print("\nsix JGSS blocks, share naming a religion, and each block over its own survey's "
          "national figure:")
    print(pd.concat([B.round(1), rel.round(2).add_suffix(" rel")], axis=1).to_string())
    for c in ["jgss_2015", "jgss_2000", "nhk_1996"]:
        print(f"    Spearman gfs_2023 vs {c}: {_sp(B['gfs_2023'], B[c]):+.3f}")


if __name__ == "__main__":
    main()
