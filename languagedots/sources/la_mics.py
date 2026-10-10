"""Laos: language retention by ethnic group from LSIS III (MICS6) 2023 microdata, applied to the
2015 census village counts -> data/normalized/la_mics.csv (and la_mics_shares.csv, the shares).

    python sources/la_mics.py          (after sources/la_census.py, whose la.csv it reads)

SOURCE. Lao Social Indicator Survey III 2023 (MICS6; Lao Statistics Bureau, Ministry of Health,
UNICEF), SPSS files from mics.unicef.org (Anita's UNICEF account, 2026-10-09), unpacked to
data/raw/la/mics_2023/ (gitignored; research use, no redistribution). 20,993 households sampled,
20,325 interviewed, in 18 provinces (HH7).

WHAT MICS CAN SAY HERE. The census asks ethnic group, not language, so every member of a group was
drawn on the group's language. MICS has the ethnic group of the household head (HC2: the census's
49-group list in the same order, plus Bu) and these language items, all answered Lao / other:
  HH16, FS14, WM14, MWM14  native language of the respondent (household, child's caretaker,
                           woman, man)
  HH15, FS13, ...          language of the interview
  FL7                      language the child (7-14) speaks most of the time at home
                           (fs.sav, foundational learning module; one child per household)

WHY FL7 AND NOT HH16. HH16 follows the interview language: of 343 households interviewed in
another language, 338 answer HH16 other; of 19,982 interviewed in Lao, 18,269 answer Lao.
Hmong-headed households interviewed in Lao without a translator answer Lao 90% of the time, those
interviewed in another language 1%. So HH16 has 86% of the Hmong, 91% of the Khmu and 100% of the
Bru (Makong) Lao-native, which nobody believes. The same households' children say otherwise: of
the 2,067 children who speak another language at home (FL7), 1,524 have a caretaker recorded
Lao-native (FS14). WM14 and MWM14 behave like HH16. FL7 asks about someone else (the child) and
about use, and gives Hmong 20% Lao at home, Khmu 33%, Akha 19%. It is the item used. It measures
children 7-14; their parents have shifted less, so applying it to everyone overstates the move.

FL7'S OWN TRAP: INTERVIEWERS. In clusters where the child's census category (Khmuic, Katuic, ...)
heads 90% or more of households, the 54 interviewers with 10+ such children record from 0% Lao
at home (seven of them) to 100% (one, 43 children in 17 clusters); in the 201 clusters where two
interviewers each saw 2+ such children, their shares differ by 41 points on average. So the model
below carries an effect per interviewer (FSINT), shrunk like a random effect, and the drawn
shares are those of a typical interviewer (effect zero).

WHICH GROUPS MOVE. Only the non-Tai groups (census codes 9-49: Mon-Khmer, Hmong-Mien,
Sino-Tibetan). Lao is Lao. The Tai groups (Tai, Phu Thai, Lue, Nhuan, Yang, Saek, Tai Nua) are not
moved: the answer list has only Lao and other, and a Phu Thai or Lue speaker may well call their
language Lao (FL7 gives them about 90% "Lao"). "Other, not stated and foreigners" is not moved.

THE MODEL. Children of the moved groups, weighted (fsweight, normalised), logistic:
    logit P(Lao at home) = f_family + d_group + b * own + u_interviewer
own = the child's census category's share of its cluster's interviewed households. Shift is
strongest where a group is a small minority: 66% Lao at home where the category heads under a
quarter of the cluster, 23% where it heads all of it. f_family is free (Mon-Khmer, Hmong-Mien,
Sino-Tibetan: MICS's own grouping); d_group has a ridge penalty LAMBDA_GROUP (a normal prior, sd
about 0.6 logit), so a thinly sampled group leans on its family and an unsampled one is its
family; u_interviewer has LAMBDA_INTERVIEWER (prior sd 1). No province term: interviewers work
inside one province's team, so the two cannot be told apart.

APPLYING IT. Each census village v gets p_v = expit(c_g + b * share_v), share_v being the
group's category's share of the village's 2015 population (the same quantity as own, on the
census). For groups with MIN_CHILDREN+ children, c_g is solved so the group's national
census-weighted mean of p_v equals the model's mean over its MICS children at interviewer effect
zero: own is measured differently on the two sides (2015 village persons, 2023 sampled cluster
households; Akha 0.83 against 0.95), so MICS sets each group's total and the slope only places
it. Thinner groups use c_g = f + d_g directly. round(people of g in v x p_v) move from g's row to a row "Lao-speaking <g>" (mapped to
Lao); integer remainders by largest remainder within each group and province. Village totals are
the census's (6,481,482).

CHECKS (asserted). File sizes; HH16 follows HH15 (the reason above); the province crosswalk (MICS
HH7 codes = the census's PCODE: per province the non-Tai share of MICS persons and of the census
within 15 points, correlation above 0.9); slope inside -3.5..-1; calibration to 1e-6; village
totals unchanged; national Lao after the move inside 52-70%.
"""
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
import la_census as LC  # noqa: E402

RAW = HERE / "data" / "raw" / "la" / "mics_2023"
IN = HERE / "data" / "normalized" / "la.csv"
OUT = HERE / "data" / "normalized" / "la_mics.csv"
SHARES = HERE / "data" / "normalized" / "la_mics_shares.csv"
SOURCE_ID = "lsis3_2023_fl7"
N_HH, N_INT = 20_993, 20_325
CENSUS_VILLAGE_TOTAL = 6_481_482
LAMBDA_GROUP = 3.0
LAMBDA_INTERVIEWER = 1.0
MIN_CHILDREN = 20           # groups with fewer FL7 children are not calibrated
SHIFT = "Lao-speaking "     # label prefix of the moved rows (taxonomy/la2015.py maps it to Lao)

PROV = {1: "Vientiane Capital", 2: "Phongsaly", 3: "Luang Namtha", 4: "Oudomxay", 5: "Bokeo",
        6: "Luang Prabang", 7: "Huaphanh", 8: "Xayabury", 9: "Xiengkhuang", 10: "Vientiane",
        11: "Borikhamxay", 12: "Khammouane", 13: "Savannakhet", 14: "Saravane", 15: "Sekong",
        16: "Champasack", 17: "Attapeu", 18: "Xaysomboun"}
FAMILY = {"khmuic": "Mon-Khmer", "palaungic": "Mon-Khmer", "katuic": "Mon-Khmer",
          "bahnaric_khmer": "Mon-Khmer", "vietic": "Mon-Khmer", "hmong": "Hmong-Mien",
          "mien": "Hmong-Mien", "tibeto_burman": "Sino-Tibetan"}
FAMILIES = sorted(set(FAMILY.values()))
MOVED = set(range(9, 50))           # census / HC2 codes that are moved (not Lao, not Tai)
TAI = set(range(1, 9))
CODE_OF = {n: c for c, n, _ in LC.P27}
CODE_OF.update({"Phong (Khmuic category)": 12, "Phong (Vietic category)": 12})
CAT_OF_LABEL = {"Phong (Khmuic category)": "khmuic", "Phong (Vietic category)": "vietic"}
FL7_LAO, FL7_OTHER = 11, 96
LAO, OTHER = 1, 6


def expit(x):
    return 1 / (1 + math.exp(-x))


def wshare(d, col="lao", w="fsweight"):
    return float((d[col] * d[w]).sum() / d[w].sum())


def load():
    import pyreadstat
    hh, _ = pyreadstat.read_sav(str(RAW / "hh.sav"), usecols=[
        "HH1", "HH2", "HH6", "HH7", "HH15", "HH16", "HH17", "HC2", "hhweight"])
    hl, _ = pyreadstat.read_sav(str(RAW / "hl.sav"), usecols=["HH1", "HH2"])
    fs, _ = pyreadstat.read_sav(str(RAW / "fs.sav"), usecols=[
        "HH1", "HH2", "FSINT", "FL7", "FS14", "fsweight"])
    if len(hh) != N_HH or int((hh.hhweight > 0).sum()) != N_INT:
        raise SystemExit(f"hh.sav: {len(hh)} rows, {int((hh.hhweight > 0).sum())} interviewed")
    hh = hh[hh.hhweight > 0].copy()
    size = hl.groupby(["HH1", "HH2"]).size().rename("members").reset_index()
    hh = hh.merge(size, on=["HH1", "HH2"], how="left", validate="1:1")
    if hh["members"].isna().any():
        raise SystemExit("interviewed households with no members in hl.sav")
    hh["pw"] = hh["hhweight"] * hh["members"]
    hh["cat"] = hh["HC2"].map(lambda c: LC.GROUP_CATEGORY.get(int(c), "other")
                              if c == c else "other")
    fs = fs[(fs.fsweight > 0) & fs.FL7.isin([FL7_LAO, FL7_OTHER])].merge(
        hh[["HH1", "HH2", "HH7", "HC2", "cat"]], on=["HH1", "HH2"], how="left",
        validate="m:1")
    if fs["HC2"].isna().any():
        raise SystemExit("children whose household is not in hh.sav")
    fs["lao"] = (fs.FL7 == FL7_LAO).astype(float)
    cl = hh.groupby(["HH1", "cat"]).size().unstack(fill_value=0)
    cl = cl.div(cl.sum(axis=1), axis=0)
    fs["own"] = [cl.at[h, c] for h, c in zip(fs.HH1, fs.cat)]
    print(f"  {len(hh):,} interviewed households, {int(hh.members.sum()):,} members; "
          f"{len(fs):,} children 7-14 with a Lao/other answer to FL7")
    return hh, fs


def check_hh16(hh, fs):
    """Why HH16 is not used: it follows the interview language."""
    t = hh.groupby(["HH15", "HH16"]).size()
    lao_lao, lao_oth = int(t.get((LAO, LAO), 0)), int(t.get((LAO, OTHER), 0))
    oth_lao, oth_oth = int(t.get((OTHER, LAO), 0)), int(t.get((OTHER, OTHER), 0))
    print(f"  HH15 interview language x HH16 native language (households): Lao/Lao {lao_lao:,}, "
          f"Lao/other {lao_oth:,}, other/Lao {oth_lao}, other/other {oth_oth}")
    if oth_oth / (oth_lao + oth_oth) < 0.95:
        raise SystemExit("HH16 no longer follows HH15; revisit the choice of item")
    hm = hh[hh.HC2 == 41]
    a = hm[(hm.HH15 == LAO) & (hm.HH17 == 3)]
    b = hm[hm.HH15 == OTHER]
    print(f"  Hmong-headed households answering HH16 Lao: {(a.HH16 == LAO).mean() * 100:.0f}% of "
          f"{len(a)} interviewed in Lao without a translator, {(b.HH16 == LAO).mean() * 100:.0f}% "
          f"of {len(b)} interviewed in another language")
    x = fs[fs.FL7 == FL7_OTHER]
    print(f"  children speaking another language at home (FL7): {len(x):,}, of whom "
          f"{int((x.FS14 == LAO).sum()):,} have a caretaker recorded Lao-native (FS14)")
    hh = hh.assign(l16=(hh.HH16 == LAO).astype(float))
    gap = {}
    rows = []
    for c in sorted(MOVED | TAI | {1}):
        h, f = hh[hh.HC2 == c], fs[fs.HC2 == c]
        if len(f) < 30:
            continue
        p16, p7 = wshare(h, "l16", "pw") * 100, wshare(f) * 100
        gap[c] = p16 - p7
        rows.append((LC.NAME[c], p16, p7, len(f), f.HH1.nunique()))
    if gap[41] < 40:
        raise SystemExit("Hmong HH16 and FL7 now agree; revisit the choice of item")
    return rows


def check_interviewers(fs):
    m = fs[fs.HC2.isin(MOVED) & (fs.own >= 0.9)]
    iv = m.groupby("FSINT").lao.agg(["size", "mean"])
    iv = iv[iv["size"] >= 10]
    w = m.groupby(["HH1", "FSINT"]).lao.agg(["size", "mean"]).reset_index()
    w = w[w["size"] >= 2]
    multi = w.groupby("HH1").filter(lambda d: len(d) >= 2)
    spread = multi.groupby("HH1")["mean"].agg(lambda s: s.max() - s.min())
    print(f"  interviewers with 10+ minority children in clusters 90%+ their own category: "
          f"{len(iv)}, Lao at home from {iv['mean'].min() * 100:.0f}% to "
          f"{iv['mean'].max() * 100:.0f}% ({int((iv['mean'] == 0).sum())} at 0%, "
          f"{int((iv['mean'] >= 0.8).sum())} at 80%+); {len(spread)} clusters with two such "
          f"interviewers differ by {spread.mean() * 100:.0f} points on average")


def check_crosswalk(hh, census):
    """MICS HH7 code = census PCODE: compare each province's non-Tai share."""
    import numpy as np
    m = {int(p): d.loc[d.HC2.isin(MOVED), "pw"].sum() / d.pw.sum() * 100
         for p, d in hh.groupby("HH7")}
    c = defaultdict(lambda: [0, 0])
    for r in census:
        c[r["prov"]][1] += r["count"]
        if CODE_OF.get(r["source_category"]) in MOVED:
            c[r["prov"]][0] += r["count"]
    c = {p: a / b * 100 for p, (a, b) in c.items()}
    if set(m) != set(PROV) or set(c) != set(PROV):
        raise SystemExit(f"provinces: MICS {sorted(m)}, census {sorted(c)}")
    bad = [PROV[p] for p in PROV if abs(m[p] - c[p]) > 15]
    r = np.corrcoef([m[p] for p in PROV], [c[p] for p in PROV])[0, 1]
    print(f"  province crosswalk: non-Tai share of persons, MICS 2023 vs census 2015, "
          f"correlation {r:.3f}, largest gap "
          + max((f"{PROV[p]} {m[p]:.1f} vs {c[p]:.1f}" for p in PROV),
                key=lambda s: abs(float(s.split()[-3]) - float(s.split()[-1]))))
    if bad or r < 0.9:
        raise SystemExit(f"crosswalk check failed: {bad}, r={r:.3f}")


def fit(fs):
    """The logistic model in the module docstring. Returns {group: f + d}, {family: f}, b."""
    import numpy as np
    d = fs[fs.HC2.isin(MOVED)].reset_index(drop=True)
    groups = sorted(int(g) for g in d.HC2.unique())
    ivs = sorted(d.FSINT.unique())
    nf, ng, ni = len(FAMILIES), len(groups), len(ivs)
    X = np.zeros((len(d), nf + ng + 1 + ni))
    fam = d.cat.map(FAMILY)
    for j, f in enumerate(FAMILIES):
        X[:, j] = (fam.values == f)
    for j, g in enumerate(groups):
        X[:, nf + j] = (d.HC2.values == g)
    X[:, nf + ng] = d.own.values
    for j, i in enumerate(ivs):
        X[:, nf + ng + 1 + j] = (d.FSINT.values == i)
    y = d.lao.values
    w = d.fsweight.values / d.fsweight.mean()
    pen = np.r_[np.full(nf, 1e-6), np.full(ng, LAMBDA_GROUP), 0.0,
                np.full(ni, LAMBDA_INTERVIEWER)]
    P = np.diag(pen)
    beta = np.zeros(X.shape[1])
    for _ in range(200):
        p = 1 / (1 + np.exp(-(X @ beta)))
        H = X.T @ ((w * p * (1 - p))[:, None] * X) + P
        step = np.linalg.solve(H, X.T @ (w * (y - p)) - P @ beta)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    else:
        raise SystemExit("model did not converge")
    f = dict(zip(FAMILIES, beta[:nf]))
    b = float(beta[nf + ng])
    u = beta[nf + ng + 1:]
    a = {g: f[FAMILY[LC.GROUP_CATEGORY[g]]] for g in MOVED}
    for j, g in enumerate(groups):
        a[g] += beta[nf + j]
    print(f"  model: {len(d):,} children of the moved groups, {ng} groups, {ni} interviewers; "
          f"slope on own-category share {b:.2f}; interviewer effects sd {u.std():.2f} "
          f"(5-95%: {np.percentile(u, 5):.2f} to {np.percentile(u, 95):.2f})")
    for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, .9), (.9, .999), (.999, 1.01)):
        x = d[(d.own >= lo) & (d.own < hi)]
        print(f"    own share {lo:.2f}-{min(hi, 1):.2f}: {len(x):>5} children, Lao at home "
              f"{wshare(x) * 100:5.1f}%")
    if not -3.5 <= b <= -1.0:
        raise SystemExit("slope outside -3.5..-1")
    return a, f, b, d


def read_census():
    rows = []
    with open(IN, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["count"] = int(r["count"])
            r["prov"] = int(r["note"].split("province=")[1].split(";")[0])
            rows.append(r)
    if sum(r["count"] for r in rows) != CENSUS_VILLAGE_TOTAL:
        raise SystemExit("la.csv total changed; re-run sources/la_census.py and check")
    if any(r["source_category"].startswith(SHIFT) for r in rows):
        raise SystemExit("la.csv already holds shifted rows")
    return rows


def calibrate(cells, a, b, d):
    """For groups with MIN_CHILDREN+ children: the intercept that makes the group's national
    census-weighted share equal the model's share over its MICS children at a typical
    interviewer. Census villages and MICS clusters measure `own` differently (persons in a
    2015 village; sampled households in a 2023 cluster), so the slope places, MICS sets totals."""
    by_g = defaultdict(list)
    for (g, p), rs in cells.items():
        by_g[g] += rs
    out, info = dict(a), {}
    for g, rs in by_g.items():
        x = d[d.HC2 == g]
        n = sum(r["count"] for r in rs)
        own_c = sum(r["count"] * r["own"] for r in rs) / n
        if len(x) < MIN_CHILDREN:
            info[g] = dict(target=None, own_mics=None, own_census=own_c)
            continue
        target = float(sum(w * expit(a[g] + b * o) for w, o in zip(x.fsweight, x.own))
                       / x.fsweight.sum())

        def mean(c):
            return sum(r["count"] * expit(c + b * r["own"]) for r in rs) / n
        lo, hi = -30.0, 30.0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if mean(mid) < target else (lo, mid)
        out[g] = (lo + hi) / 2
        if abs(mean(out[g]) - target) > 1e-6:
            raise SystemExit(f"{LC.NAME[g]}: calibration missed")
        info[g] = dict(target=target, own_mics=wshare(x, "own"), own_census=own_c)
    return out, info


def apply(rows, a, b, d):
    vpop, vcat = defaultdict(int), defaultdict(int)
    for r in rows:
        vpop[r["geo_id"]] += r["count"]
        code = CODE_OF.get(r["source_category"])
        if code in MOVED:
            r["cat"] = CAT_OF_LABEL.get(r["source_category"], LC.GROUP_CATEGORY[code])
            vcat[(r["geo_id"], r["cat"])] += r["count"]
    cells = defaultdict(list)                    # (group, province) -> its village rows
    for r in rows:
        if CODE_OF.get(r["source_category"]) in MOVED:
            r["own"] = vcat[(r["geo_id"], r["cat"])] / vpop[r["geo_id"]]
            cells[(CODE_OF[r["source_category"]], r["prov"])].append(r)
    a, info = calibrate(cells, a, b, d)
    moved = {}
    for (g, p), rs in cells.items():
        raw = [r["count"] * expit(a[g] + b * r["own"]) for r in rs]
        base = [int(x) for x in raw]
        left = round(sum(raw)) - sum(base)
        for i in sorted(range(len(rs)), key=lambda i: raw[i] - base[i], reverse=True)[:left]:
            base[i] += 1
        for r, k in zip(rs, base):
            moved[id(r)] = k
    out = []
    for r in rows:
        k = moved.get(id(r), 0)
        keep = r["count"] - k
        if keep < 0:
            raise SystemExit("moved more than the row holds")
        base = dict(geo_id=r["geo_id"], geo_level=r["geo_level"], geo_name=r["geo_name"],
                    note=r["note"])
        if keep:
            out.append(dict(base, source_category=r["source_category"], count=keep))
        if k:
            out.append(dict(base, source_category=SHIFT + r["source_category"], count=k,
                            note=r["note"] + "; LSIS III 2023 children's home language "
                                             f"(FL7) model, {k / r['count'] * 100:.1f}% of "
                                             "the group here"))
    return out, a, info


def label(lab):
    lab = "Lao" if lab.startswith(SHIFT) else lab
    return "Phong" if lab.startswith("Phong") else lab


def summarise(before, after):
    def tot(rows, key):
        t = defaultdict(int)
        for r in rows:
            t[(key(r), label(r["source_category"]))] += r["count"]
        return t
    for r in after:
        r["prov"] = int(r["note"].split("province=")[1].split(";")[0])
    n = sum(r["count"] for r in before)
    b0, a0 = tot(before, lambda r: 0), tot(after, lambda r: 0)
    print(f"  national, % of {n:,}: before -> after")
    for (_, lab), v in sorted(b0.items(), key=lambda kv: -kv[1])[:14]:
        print(f"    {lab:<34}{v / n * 100:6.2f} ->{a0[(0, lab)] / n * 100:6.2f}")
    bp, ap = tot(before, lambda r: r["prov"]), tot(after, lambda r: r["prov"])
    pn = defaultdict(int)
    for r in before:
        pn[r["prov"]] += r["count"]
    for p in PROV:
        top = sorted(((lab, v) for (q, lab), v in bp.items() if q == p), key=lambda kv: -kv[1])[:4]
        print(f"    {PROV[p]:<18}" + "; ".join(
            f"{lab} {v / pn[p] * 100:.1f}->{ap[(p, lab)] / pn[p] * 100:.1f}" for lab, v in top))
    return a0[(0, "Lao")] / n * 100


def main():
    hh, fs = load()
    hh16_rows = check_hh16(hh, fs)
    check_interviewers(fs)
    census = read_census()
    check_crosswalk(hh, census)
    a, fam, b, d = fit(fs)
    out, a, cal = apply(census, a, b, d)
    vt0, vt1 = defaultdict(int), defaultdict(int)
    for r in census:
        vt0[r["geo_id"]] += r["count"]
    for r in out:
        vt1[r["geo_id"]] += r["count"]
    if vt0 != vt1 or sum(vt1.values()) != CENSUS_VILLAGE_TOTAL:
        raise SystemExit("a village's total changed")

    ppl, mov = defaultdict(int), defaultdict(int)
    for r in census:
        g = CODE_OF.get(r["source_category"])
        if g in MOVED:
            ppl[g] += r["count"]
    for r in out:
        if r["source_category"].startswith(SHIFT):
            mov[CODE_OF[r["source_category"][len(SHIFT):]]] += r["count"]
    p16 = {n: (x, y) for n, x, y, _, _ in hh16_rows}
    print("  family intercepts (logit at own share 0): "
          + ", ".join(f"{k} {expit(v) * 100:.0f}%" for k, v in fam.items()))
    print(f"  {'group':<12}{'children':>9}{'clusters':>9}{'HH16 %':>7}{'FL7 raw %':>10}"
          f"{'model %':>8}{'drawn %':>8}{'own M/C':>10}{'people 2015':>12}{'moved':>9}")
    shares_rows = []
    for g in sorted(MOVED, key=lambda g: -ppl[g]):
        x = d[d.HC2 == g]
        raw = wshare(x) if len(x) else None
        drawn = mov[g] / ppl[g] if ppl[g] else 0
        h16 = p16.get(LC.NAME[g], (None, None))[0]
        c = cal.get(g, {})
        tgt, om, oc = c.get("target"), c.get("own_mics"), c.get("own_census")
        print(f"  {LC.NAME[g]:<12}{len(x):>9}{x.HH1.nunique():>9}"
              f"{f'{h16:.1f}' if h16 is not None else '-':>7}"
              f"{f'{raw * 100:.1f}' if raw is not None else '-':>10}"
              f"{f'{tgt * 100:.1f}' if tgt is not None else '-':>8}"
              f"{drawn * 100:>8.1f}"
              f"{(f'{om:.2f}/' if om is not None else '-/') + (f'{oc:.2f}' if oc is not None else '-'):>10}"
              f"{ppl[g]:>12,}{mov[g]:>9,}")
        shares_rows.append(dict(group=LC.NAME[g], family=FAMILY[LC.GROUP_CATEGORY[g]],
                                children=len(x), clusters=x.HH1.nunique(),
                                fl7_raw_share=round(raw, 4) if raw is not None else "",
                                model_share=round(tgt, 4) if tgt is not None else "",
                                logit_intercept=round(a[g], 4), slope=round(b, 4),
                                drawn_share=round(drawn, 4), people_2015=ppl[g],
                                moved=mov[g]))
    lao = summarise(census, out)
    if not 52 <= lao <= 70:
        raise SystemExit(f"national Lao {lao:.1f}% outside 52-70%")

    for path, rows, fields in (
            (OUT, out, ["geo_id", "geo_level", "geo_name", "source_category", "count",
                        "source_id", "note"]),
            (SHARES, shares_rows, list(shares_rows[0]))):
        tmp = path.with_suffix(".part")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(dict(r, source_id=SOURCE_ID) if path == OUT else r)
        tmp.replace(path)
    print(f"  wrote {OUT.relative_to(HERE)} ({len(out):,} rows) and "
          f"{SHARES.relative_to(HERE)} ({len(shares_rows)} rows); "
          f"{sum(mov.values()):,} people moved onto Lao")


if __name__ == "__main__":
    main()
