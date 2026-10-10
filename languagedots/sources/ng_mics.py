"""Nigeria: first language by state from MICS6 2021 microdata for the nine languages MICS names,
with MICS's "other language" split by the pooled Afrobarometer -> data/normalized/ng.csv.

    python sources/ng_mics.py

SOURCE. Nigeria Multiple Indicator Cluster Survey 2021 (MICS6, with the National Immunization
Coverage Survey; National Bureau of Statistics and UNICEF), SPSS files from mics.unicef.org
(Anita's UNICEF account, 2026-10-09), unzipped to data/raw/ng/mics_2021/ (gitignored; research
use, no redistribution of the files). 41,532 households, 39,632 interviewed (hhweight > 0, the
MICS-NICS weight), 207,496 members, 2,079 clusters, all 37 states (HH7, FCT included).

ITEM. HC1B, "Language of household head", MICS's household mother-tongue item, read as every
member's (hl.sav members x hhweight, so shares are of persons). Ten answers: Hausa, Igbo,
Yoruba, Fulani, Kanuri, Ijaw, Tiv, Ibibio, Edo, other language (25% of persons nationally, and
most of the Middle Belt and the South-South).

WHY HC1B AND NOT HH16 (respondent's native language) OR WM14 (women 15-49). HC1B and HH16 differ
in 2,609 of 39,632 households. In 24,072 the respondent IS the head, so both items describe one
person, and they still differ in 1,419. Those same-person differences follow the interview
language (HH15): Kanuri heads interviewed in Hausa give HH16 Hausa 60 times in 335 and never
otherwise; heads interviewed in Yoruba give HH16 Yoruba though their language is Igbo (23 of 56),
Hausa (18 of 54), Fulani (20 of 82), Tiv (8 of 21), Edo (9 of 16), other (73 of 297); in English
interviews the two items almost always agree. So HH16 slides towards the interview language, as
it did in Iraq (sources/iq_mics6.py), and HC1B is drawn. HC1B tracks the head's ethnic group (HC2)
in 96% of households, but not blindly: 246 Edo and 215 Ijaw heads by ethnicity give "other
language" (Esan, Yekhee, Kalabari, Okrika...), so the item is answered as language.
  The call that matters is Fulani. Nationally: HC1B 7.2% of persons, HH16 5.4%, WM14 5.0%; in
Bauchi 29.7 / 18.6 / 19.3, Gombe 49.8 / 36.6 / 34.0, Kano 17.2 / 12.8 / 12.5. A Fulani head who
answers HH16 Hausa (322 of 1,545 same-person cases) may be one whose first language really is
Hausa, so HC1B is an upper reading for Fulfulde; it is drawn, and the record says so.

HOW MICS ENTERS (per state):
  1. The nine named MICS answers' shares are drawn as measured: about 1,000 households and 56 clusters
     a state against the Afrobarometer's median 280 respondents. Each is matched to the
     Afrobarometer answer of the same name only (MICS's Ijaw is Afrobarometer's Ijaw, not
     Kalabari or Okrika; its Ibibio not Efik or Anaang; its Igbo not Ika or Ikwerre; its Edo not
     Esan or Urhobo): checked state by state, e.g. Rivers Ijaw MICS 3.0% / Afrobarometer Ijaw
     3.6% (with Kalabari, Okrika, Nembe, Ibani 29.5%), Delta Edo 1.7 / 1.9 (Edoid 42.0).
  2. MICS's "other language" is split across the Afrobarometer's other answers in the state
     (pooled R4-R9 home language, after sources/ng_afro.py's English and Pidgin moves, its
     verbatim reading and R4's combined label), with K_OTHER pseudo-respondents on "Other
     Nigerian language": weight of answer a = (Afrobarometer weighted respondents naming a +
     K_OTHER x [a is Other Nigerian language]) / (all its other-answer respondents + K_OTHER).
     Where the Afrobarometer has many such respondents (Plateau 179, Kogi 176, Rivers 359) its
     split stands; where it has a handful against a real MICS share (Oyo: 3 respondents, MICS
     6.9%, mostly rural Christian heads of other ethnic groups, plausibly migrant farmers), most
     of the share is drawn as an unnamed Nigerian language rather than multiplying three people.
  3. English and Nigerian Pidgin, which HC1B does not offer, keep the current build's levels
     (Anita's rulings): English at the Afrobarometer R7 mother-tongue share, shrunk as in
     ng_afro.state_shares; Pidgin at Ethnologue's 4.7 million, spread by ng_afro.pidgin_pattern.
     Every other answer is scaled to what is left.
  The current build's "Hausa outside Hausaland" step (R7 mother-tongue shrink, ask 018) is
  dropped: HC1B is itself a mother-tongue item, so MICS's Hausa replaces it everywhere.

WEIGHTS. hhweight (the MICS-NICS weight; hhweightMICS gives the same shares) varies up to 200-fold
inside a state. In Ebonyi three urban clusters carry 87% of the state's weighted people (about
4 effective clusters of 46), in Anambra about 7, Bayelsa 11, Imo 11; Ebonyi's 7.4% Hausa is
three households in two of those clusters (and 7 of its 10 Hausa-coded heads give Igbo as
ethnic group and as the respondent's language). So each household's weight is capped at TRIM
(5) times its state's median, a standard trim. It moves the MICS shares by more than a point in
four states only: Ebonyi Hausa 7.4 -> 2.6%, Igbo 92.6 -> 97.3; Bayelsa Ijaw 61.8 -> 71.3%;
Kaduna and Kwara about a point. In Ebonyi and Bayelsa it moves MICS towards the Afrobarometer
(Igbo 97.5, Ijaw 72.6), an independent sample. The cost: Bayelsa's urban share by weight goes
29 -> 9%.

ZEROS. Clusters per state: 38 (smallest) to 78. A group living in enclaves holding a share p of a
state's people is missed by every sampled cluster with probability (1 - p)^clusters: 2% of a
38-cluster state 46% of the time, 5% 14%. That bounds what a MICS zero says about the nine named
languages; the cases where the Afrobarometer had 1%+ and MICS has none are printed.

POPULATION. COD-PS 2022 state totals (religiondots' ng_lookup.csv, read-only), 216,798,930, as
before; largest-remainder rounding inside each state.

CHECKS (asserted). Row counts; every member matches a household; HH7 -> the 37 states both
ways; HC1B = HH16 in 90%+ of households; across the 37 states MICS and the Afrobarometer agree
on where Hausa, Yoruba and Igbo are (Pearson r > 0.9); shares sum to 1; drawn total = COD-PS;
Pidgin = 4.7 million.
"""
import csv
import io
import contextlib
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
import ng_afro as A  # noqa: E402

RAW = HERE / "data" / "raw" / "ng" / "mics_2021"
OUT = HERE / "data" / "normalized" / "ng.csv"
CODPS_2022 = A.CODPS_2022
N_HH, N_INT, N_HL = 41_532, 39_632, 207_496
K_OTHER = 8.0      # pseudo-respondents on "Other Nigerian language" in the split of MICS's other
TRIM = 5.0         # household weights capped at this multiple of their state's median

# HC1B answer -> the Afrobarometer answer (and ng.csv label) it is
HEAD = {"HAUSA": "Hausa", "IGBO": "Igbo", "YORUBA": "Yoruba", "FULANI": "Fula",
        "KANURI": "Kanuri", "IJAW": "Ijaw", "TIV": "Tiv", "IBIBIO": "Ibibio", "EDO": "Edo",
        "OTHER LANGUAGE": "_other"}
TEN = [v for v in HEAD.values() if v != "_other"]


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def state_of(x):
    return A.NORM.get(A.ckey(str(x)))


def mics():
    import pyreadstat
    hh, _ = pyreadstat.read_sav(str(RAW / "hh.sav"), apply_value_formats=True,
                                usecols=["HH1", "HH2", "HH7", "HH15", "HH16", "HH47", "HC1B",
                                         "HC2", "hhweight"])
    hl, _ = pyreadstat.read_sav(str(RAW / "hl.sav"), usecols=["HH1", "HH2", "HL1", "HL3"])
    wm, _ = pyreadstat.read_sav(str(RAW / "wm.sav"), apply_value_formats=True,
                                usecols=["HH7", "WM14", "wmweight"])
    say(len(hh) == N_HH and int((hh.hhweight > 0).sum()) == N_INT,
        f"hh.sav: {len(hh):,} households, {int((hh.hhweight > 0).sum()):,} interviewed")
    i = hh[hh.hhweight > 0].copy()
    say(i["HC1B"].notna().all() and i["HH16"].notna().all(),
        "every interviewed household has HC1B and HH16")
    for col in ("HC1B", "HH16"):
        bad = sorted(set(i[col].astype(str)) - set(HEAD))
        say(not bad, f"{col}: every answer is one of the ten ({bad})")
    i["state"] = i["HH7"].map(state_of)
    say(i["state"].notna().all() and i["state"].nunique() == A.N_STATES,
        f"HH7 -> {i['state'].nunique()} states")

    # weight trim (see WEIGHTS in the docstring)
    cap = i.groupby("state")["hhweight"].transform("median") * TRIM
    i["w"] = np.minimum(i["hhweight"], cap)
    hlm = hl.groupby(["HH1", "HH2"]).size().rename("nmem").reset_index()
    i = i.merge(hlm, on=["HH1", "HH2"], how="left")

    def eff_clusters(wcol):
        c = (i[wcol] * i["nmem"]).groupby([i["state"], i["HH1"]]).sum()
        g = c.groupby(level=0)
        return g.sum() ** 2 / (c ** 2).groupby(level=0).sum()
    e0, e1 = eff_clusters("hhweight"), eff_clusters("w")
    trimmed = i[i["w"] < i["hhweight"]]
    print(f"  weights trimmed at {TRIM:g}x the state median: {len(trimmed):,} households in "
          f"{trimmed['state'].nunique()} states; effective clusters (persons), before -> after: "
          + ", ".join(f"{s} {e0[s]:.1f} -> {e1[s]:.1f}" for s in e0.sort_values().index[:5]))
    say(e1.min() >= 10, f"after the trim every state has 10+ effective clusters "
        f"(lowest {e1.idxmin()} {e1.min():.1f})")

    persons = hl.merge(i[["HH1", "HH2", "state", "HC1B", "w"]], on=["HH1", "HH2"],
                       how="inner")
    say(len(persons) == N_HL and len(hl) == N_HL,
        f"hl.sav: {len(hl):,} members, all in interviewed households")

    # --- item comparison
    agree = (i["HC1B"].astype(str) == i["HH16"].astype(str))
    say(agree.mean() >= 0.90, f"HC1B = HH16 in {agree.sum():,} of {len(i):,} households")
    head = hl[hl["HL3"] == 1][["HH1", "HH2", "HL1"]].rename(columns={"HL1": "head_line"})
    j = i.merge(head, on=["HH1", "HH2"], how="left")
    same = j[j["HH47"] == j["head_line"]]
    print(f"  respondent is the head in {len(same):,}; HC1B != HH16 there in "
          f"{int((same.HC1B.astype(str) != same.HH16.astype(str)).sum()):,}")
    print("  same person, HH16 by interview language (HH15) for heads whose HC1B is not that "
          "language:")
    for lang, hh15 in (("HAUSA", "HAUSA"), ("YORUBA", "YORUBA")):
        g = same[(same.HH15.astype(str) == hh15) & (same.HC1B.astype(str) != lang)]
        n = g.groupby(g.HC1B.astype(str)).size()
        k = g[g.HH16.astype(str) == lang].groupby(g.HC1B.astype(str)).size()
        print(f"    {hh15.title()} interviews -> HH16 {lang.title()}: " + ", ".join(
            f"{h.title()} {int(k.get(h, 0))}/{int(v)}" for h, v in n.items() if v >= 15))
        g = same[(same.HH15.astype(str) == "ENGLISH") & (same.HC1B.astype(str) != lang)]
        print(f"    English interviews -> HH16 {lang.title()}: "
              f"{int((g.HH16.astype(str) == lang).sum())}/{len(g)}")
    ct = pd.crosstab(i["HC1B"].astype(str), i["HC2"].astype(str))
    print(f"  HC1B = HC2 (ethnic group) in {int(np.trace(ct.reindex(columns=ct.index).fillna(0).to_numpy())):,}"
          f" of {len(i):,}; Edo heads by ethnicity giving other language "
          f"{int(ct.loc['OTHER LANGUAGE', 'EDO'])}, Ijaw {int(ct.loc['OTHER LANGUAGE', 'IJAW'])}")

    # --- persons by state x HC1B; HH16 and WM14 as other readings
    p = persons.groupby(["state", "HC1B"], observed=True)["w"].sum()
    p = p.div(p.groupby(level="state").sum(), level="state").unstack(fill_value=0.0)
    p.columns = [HEAD[str(c)] for c in p.columns]
    i["pw"] = i["hhweight"] * i["nmem"]     # national readings: the survey's own weights
    wm = wm[wm["wmweight"] > 0].copy()
    wm["state"] = wm["HH7"].map(state_of)
    readings = {}
    for name, d, col, w in (("HC1B", i, "HC1B", "pw"), ("HH16", i, "HH16", "pw"),
                            ("WM14", wm, "WM14", "wmweight")):
        readings[name] = d.groupby(d[col].astype(str))[w].sum() / d[w].sum()
    print("  national, % of persons (WM14: of women 15-49):  HC1B | HH16 | WM14")
    for a in HEAD:
        print(f"    {HEAD[a]:8s} " + " | ".join(f"{readings[k].get(a, 0) * 100:5.2f}"
                                               for k in ("HC1B", "HH16", "WM14")))
    clusters = i.groupby("state")["HH1"].nunique()
    n_hh = i.groupby("state").size()
    print(f"  clusters per state: {clusters.min()} ({clusters.idxmin()}) to {clusters.max()} "
          f"({clusters.idxmax()}), median {int(clusters.median())}; households per state "
          f"{n_hh.min()} to {n_hh.max()}")
    return p, clusters, n_hh


def afro():
    """The pooled Afrobarometer, harmonised as sources/ng_afro.py does (English and Pidgin
    answers moved to the respondent's ethnic group's language, verbatims read, R4's combined
    label shared), plus the English and Pidgin levels of the current build."""
    with contextlib.redirect_stdout(io.StringIO()):
        a = A.load()
        pid = A.pidgin_pattern(a)
        a = A.english_by_ethnicity(a, "English")
        a = A.english_by_ethnicity(a, "Nigerian Pidgin")
        A.single_verbatims(a)
        a = A.split_combined(a)
    rest = a[~a["answer"].isin(A.ENGLISH_PIDGIN)]
    w = rest.groupby(["state", "answer"])["w"].sum().unstack(fill_value=0.0)
    from wafr_afro import r7_mother, shrink
    t = r7_mother("Nigeria", {"English": ["English"]})
    reg = {g: A.NORM[A.ckey(g)] for g in t.index if g != "_national"}
    eng = shrink(t, "English").rename(reg)
    return w, eng, pid


def main():
    p, clusters, n_hh = mics()
    w, eng, pid = afro()
    lut = pd.read_csv(A.LOOKUP, dtype={"geo_id": str})
    say(len(lut) == A.N_STATES and int(lut["pop"].sum()) == CODPS_2022,
        f"ng_lookup.csv: {len(lut)} states, {int(lut['pop'].sum()):,} people (COD-PS 2022)")
    say(sorted(p.index) == sorted(lut["name"]) == sorted(w.index),
        "MICS, the Afrobarometer and COD-PS have the same 37 states")
    pop = lut.set_index("name")["pop"].astype(float)
    gid = dict(zip(lut["name"], lut["geo_id"]))

    # does MICS put the big languages where the Afrobarometer does?
    abs_ = w.div(w.sum(axis=1), axis=0)
    print("  MICS against the Afrobarometer, Pearson r across the 37 states:")
    for lang in TEN:
        r = np.corrcoef(p[lang], abs_.get(lang, 0).reindex(p.index, fill_value=0))[0, 1]
        print(f"    {lang:8s} r = {r:+.3f}   national MICS {(p[lang] * pop).sum() / pop.sum():6.2%}"
              f"  Afrobarometer {(abs_.get(lang, 0) * pop).sum() / pop.sum():6.2%}")
        if lang in ("Hausa", "Yoruba", "Igbo"):
            say(r > 0.9, f"{lang}: the two surveys agree on where it is")
    miss = [(st, lang, abs_.loc[st, lang]) for st in p.index for lang in TEN
            if lang in abs_.columns and p.loc[st, lang] == 0 and abs_.loc[st, lang] >= 0.01]
    print("  named by 1%+ of a state's Afrobarometer respondents, none in MICS (drawn as MICS "
          "says): " + "; ".join(f"{st} {lang} {v:.1%} ({clusters[st]} clusters)"
                                for st, lang, v in miss))

    # MICS ten + other split by the Afrobarometer's other answers
    pl1 = pid.reindex(pop.index).fillna(0.0)
    pl1 = pl1 * (A.PIDGIN_L1 / (pl1 * pop).sum())
    rows, notes = {}, {}
    print("  MICS other language split (state: MICS other %, Afrobarometer other-answer "
          "respondents, share left unnamed):")
    for st in p.index:
        s = {lang: p.loc[st, lang] for lang in TEN if p.loc[st, lang] > 0}
        other = p.loc[st, "_other"]
        cand = w.loc[st].drop([c for c in TEN if c in w.columns])
        cand = cand[cand > 0]
        n = cand.sum()
        split = cand.copy()
        split[A.OTHER_NG] = split.get(A.OTHER_NG, 0.0) + K_OTHER
        split = split / (n + K_OTHER)
        if other > 0:
            for a, f in split.items():
                s[a] = s.get(a, 0.0) + other * f
            if other >= 0.03:
                print(f"    {st:26s} {other:6.1%}  n {n:6.1f}  unnamed "
                      f"{split[A.OTHER_NG] * other:6.1%}")
        if abs(sum(s.values()) - 1) >= 1e-9:
            raise SystemExit(f"{st}: MICS shares sum to {sum(s.values())}")
        fixed = {"English": eng[st]}
        if pl1[st] > 0:
            fixed["Nigerian Pidgin"] = pl1[st]
        k = 1 - sum(fixed.values())
        s = {a: v * k for a, v in s.items()}
        s.update(fixed)
        rows[st] = s
        notes[st] = other
    worst = max(abs(sum(s.values()) - 1) for s in rows.values())
    say(worst < 1e-9, f"every state's shares sum to 1 (worst {worst:.1e})")

    out, natl = [], {}
    for st in sorted(rows):
        s = rows[st]
        raw = {a: v * pop[st] for a, v in s.items()}
        cnt = {a: int(np.floor(x)) for a, x in raw.items()}
        for a in sorted(raw, key=lambda x: raw[x] - cnt[x], reverse=True)[
                :int(round(pop[st])) - sum(cnt.values())]:
            cnt[a] += 1
        assert sum(cnt.values()) == int(pop[st])
        for a in sorted(cnt, key=cnt.get, reverse=True):
            if not cnt[a]:
                continue
            natl[a] = natl.get(a, 0) + cnt[a]
            if a in TEN:
                src, note = "mics6_2021_hc1b", (f"MICS6 2021 HC1B {p.loc[st, a]:.4f} of persons "
                                                f"in {n_hh[st]} households, {clusters[st]} "
                                                f"clusters")
            elif a == "English":
                src, note = "afrobarometer_r7_q2a", f"R7 mother tongue, shrunk: {s[a]:.4f}"
            elif a == "Nigerian Pidgin":
                src, note = "ethnologue_2023", f"4.7M L1 spread by R7-R9 home answers: {s[a]:.4f}"
            else:
                src, note = ("mics6_2021_hc1b+afrobarometer",
                             f"MICS6 other language {notes[st]:.4f}, split by the Afrobarometer "
                             f"R4-R9 other answers in the state")
            out.append(dict(geo_id=gid[st], geo_level="state", geo_name=st,
                            source_category=a, count=cnt[a], tier="modelled", source_id=src,
                            year="2021", note=note))
    say(sum(natl.values()) == CODPS_2022, f"drawn total {sum(natl.values()):,}")
    say(abs(natl.get("Nigerian Pidgin", 0) - A.PIDGIN_L1) < 50,
        f"Pidgin {natl.get('Nigerian Pidgin', 0):,}")

    tmp = OUT.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["geo_id", "geo_level", "geo_name", "source_category",
                                           "count", "tier", "source_id", "year", "note"])
        wr.writeheader()
        wr.writerows(out)
    tmp.replace(OUT)
    print(f"  wrote {OUT.relative_to(HERE)}: {len(out)} rows, {len(natl)} answers, "
          f"{sum(natl.values()):,} people")
    for a, v in sorted(natl.items(), key=lambda kv: -kv[1])[:25]:
        print(f"    {a:28s} {v:>12,}  {v / CODPS_2022:6.2%}")


if __name__ == "__main__":
    main()
