"""Taiwan: the drawn folk-religion and no-religion answers split by a religious home altar.
Spec §3.13, Anita 2026-09-15. Called from `sources/tw.py` main(), after the county shares are fitted
and before `tw.csv` is written; it moves no person between counties and no count between any
other answers.

WHY TAIWAN'S FOLK RELIGION NEEDS SPLITTING AT ALL. The long card's folk answer is four interviewer
codes, and one of them is almost all of it (2018 report, pp. 103-104, weighted here):

    021 自己認為的, volunteers "folk religion"                          2.40% (2014)   1.46% (2018)
    022 拜神的, worships the gods, holds incense, or "no religion but
        worships along with my family"                               42.72%         45.92%

So the 48% drawn as `chinesefolk` before this was mostly people who did not name folk religion,
while China's 3% named it. §3.13 puts both on one rule: `chinesefolk` is folk religion named, or a
religious altar kept by someone who names nothing.

THE SPLIT, per county, of what tw.py already drew:
  * `Folk religion` becomes three answers: self-identified (code 021) at its national share of the
    folk answer in the level rounds; then the rest, 022-024, with and without a religious altar.
  * `No religious belief` becomes two: with a religious altar, and without one. The second goes to
    `unknown` like the folk-coded people without one (Anita, 2026-09-15).
The altar rates come from TSCS 2009, 2014 and 2018, the three rounds with the ISSP item
(`havshrin`, `v27`, `v55`), and are county rates only where `folk_altar.rate_test` passes; otherwise
national. The 021 share is national, because it is about thirty respondents a round.

`Buddhism: Buddha worship` (031, about 11%) is the Buddhist twin of 022 and is NOT split: those
people named Buddhism, and Buddhism is drawn from what people name.
"""

import os

import numpy as np
import pandas as pd

import folk_altar

ALTAR_VAR = {2009: "havshrin", 2014: "v27", 2018: "v55"}
SELF_ID_CODE = 21

FOLK_SELF = "Folk religion, self-identified"
FOLK_ALTAR = "Folk religion, worships the gods, religious altar at home"
FOLK_NO_ALTAR = "Folk religion, worships the gods, no religious altar"
NONE_ALTAR = "No religious belief, religious altar at home"
# On `unknown` with FOLK_NO_ALTAR, as in China and Hong Kong (Anita, 2026-09-15). It kept tw.NONE's
# name, and `unaffiliated`, for the first build that day.
NONE_NO_ALTAR = "No religious belief, no religious altar"

# Pinned after the first run (2026-09-15), as tw.py pins CARRIES. The folk rate passed (p 0.032);
# the no-religion rate missed by a hair (p 0.052, 693 respondents) and is drawn at the national 47.4%.
VERDICTS = {"folk": "own geography", "none": "not distinguishable from chance"}


def respondents(tw, zips):
    """The three altar rounds, placed and answered exactly as tw.py reads them, plus `altar`."""
    frames = []
    for year, var in ALTAR_VAR.items():
        df = tw.read_round(year, zips)
        path = os.path.join(tw.RAW, tw.ROUNDS[year]["fid"] + ".dta")
        with pd.io.stata.StataReader(path) as r:
            raw = r.read(convert_categoricals=False)
        raw.columns = [c.lower() for c in raw.columns]
        if var not in raw.columns:
            raise SystemExit(f"{tw.ROUNDS[year]['fid']}: no altar column {var}")
        df["altar"] = pd.to_numeric(raw[var], errors="coerce").loc[df.index].to_numpy()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _rate(sub, units, label):
    sub = sub[sub["altar"].isin([1, 2])]
    uidx = sub["unit"].map({u: i for i, u in enumerate(units)}).to_numpy()
    if np.isnan(uidx.astype(float)).any():
        raise SystemExit(f"{label}: respondents outside the drawn counties")
    uidx = uidx.astype(int)
    hit = (sub["altar"] == 1).to_numpy()
    t = folk_altar.rate_test(hit, uidx, sub["cluster"].to_numpy(), len(units),
                             group=sub["year"].to_numpy())
    k = np.bincount(uidx, weights=hit.astype(float), minlength=len(units))
    n = np.bincount(uidx, minlength=len(units)).astype(float)
    w = sub["w"].to_numpy()
    kw = np.bincount(uidx, weights=w * hit, minlength=len(units))
    nw = np.bincount(uidx, weights=w, minlength=len(units))
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_w = np.where(nw > 0, kw / nw, np.nan)
    national = float((w * hit).sum() / w.sum())
    if t["verdict"] == "own geography":
        share, M = folk_altar.shrink(k, n, rate_w, national)
        how = f"county rate shrunk toward national, prior strength {M:.1f} respondents"
    else:
        share = np.full(len(units), national)
        how = f"national rate ({t['verdict']})"
    print(f"    {label}: {t['n']:,} respondents, {t['k']:,} with an altar, {t['clusters']} townships; "
          f"median Spearman {t['median']:+.3f}, null 95th {t['q95']:+.3f}, p {t['p']:.4f}, "
          f"chi-square p {t['chi_p']:.2e}, largest township {t['cell']:.0%}, top county's largest "
          f"{t['top_cell']:.0%} -> {t['verdict']}; national {national:.1%}; {how}")
    return (pd.Series(share, index=units), pd.Series(rate_w, index=units),
            pd.Series(n, index=units), t["verdict"], national)


def apply(out, tw, zips, units):
    print("\n  spec §3.13: folk religion and no religion split by a religious home altar "
          f"(TSCS {', '.join(map(str, ALTAR_VAR))}):")
    r = respondents(tw, zips)

    folk = r[r["answer"] == tw.FOLK]
    lvl = folk[folk["year"].isin(tw.LEVEL_ROUNDS)]
    self_share = float((lvl["w"] * (lvl["code"] == SELF_ID_CODE)).sum() / lvl["w"].sum())
    print(f"    code 021 (names folk religion) is {self_share:.1%} of the folk answer in "
          f"{' and '.join(map(str, tw.LEVEL_ROUNDS))}, weighted; national everywhere")

    a, a_raw, a_n, va, a_nat = _rate(folk[folk["code"] != SELF_ID_CODE], units,
                                     "folk religion, worships the gods (022-024)")
    b, b_raw, b_n, vb, b_nat = _rate(r[r["answer"] == tw.NONE], units, "no religious belief (010)")
    verdicts = {"folk": va, "none": vb}
    if VERDICTS is not None and verdicts != VERDICTS:
        raise SystemExit(f"the altar tests now say {verdicts}, pinned {VERDICTS}; update "
                         "tw_altar.VERDICTS, sources/tw.md and spec §3.13 deliberately")

    print(f"\n    {'county':<18}{'folk n':>7}{'raw':>7}{'drawn':>7}{'none n':>8}{'raw':>7}{'drawn':>7}")
    for u in units:
        print(f"    {tw.UNITS[u][1]:<18}{int(a_n[u]):>7}{a_raw[u] * 100:6.1f}%{a[u] * 100:6.1f}%"
              f"{int(b_n[u]):>8}{b_raw[u] * 100:6.1f}%{b[u] * 100:6.1f}%")

    rows = []
    for d in out.to_dict("records"):
        cat, c = d["source_category"], int(d["count"])
        if cat == tw.FOLK:
            named = int(round(c * self_share))
            with_altar = int(round((c - named) * a[d["geo_id"]]))
            parts = [(FOLK_SELF, named), (FOLK_ALTAR, with_altar),
                     (FOLK_NO_ALTAR, c - named - with_altar)]
        elif cat == tw.NONE:
            with_altar = int(round(c * b[d["geo_id"]]))
            parts = [(NONE_ALTAR, with_altar), (NONE_NO_ALTAR, c - with_altar)]
        else:
            rows.append(d)
            continue
        for new_cat, k in parts:
            if k > 0:
                rows.append({**d, "source_category": new_cat, "count": k,
                             "note": d["note"] + "; split by a religious home altar (TSCS 2009-2018, spec §3.13)"})
    new = pd.DataFrame(rows, columns=out.columns)
    if int(new["count"].sum()) != int(out["count"].sum()):
        raise SystemExit("the altar split changed the drawn total")
    return new
