"""South Sudan: religion by former state, from the World Bank's High Frequency South Sudan Survey.

Reads the two STATA8 zips Anita downloaded from the World Bank Microdata Library (ask 042) and
writes data/normalized/ss.csv. `sources/ss.md` is this country's record.

    data/raw/ss/SSD_2015_HFS-W1_v02_M_STATA8.zip    catalog 2778, wave 1 (2015), drawn
    data/raw/ss/SSD_2016_HFS-W2_v02_M_STATA8.zip    catalog 2777, wave 2 (2016), a witness

The files are licensed for statistical and research use, no redistribution: they stay under
data/raw/ss/ (gitignored with all of data/) and only state shares leave this script.

## WHAT IS DRAWN

Module C, item C.9, *Please specify which religion [the household head] belongs to*, a
multi-select on the card Christianity, Islam, Traditional African Religion, Judaism, Buddhism,
Hinduism, Agnostic, Atheism, Other. It is the HEAD's religion, drawn for everyone in the household
(`playbooks/dhs_mics.md`, first trap).

**Wave 1 only, six states.** Wave 1 sampled 50 enumeration areas in each of Northern and Western
Bahr el Ghazal, Lakes, and Western, Central and Eastern Equatoria, urban and rural, 3,550
households. Wave 2 went back to towns only (no rural EAs, no `urban` variable, 1,189 households)
and added urban Warrap. Pooling the waves would count the towns twice; wave 1 is the state design
and wave 2 is held out as a replicate of its urban half (`wave2_witness`).

**Jonglei, Unity, Upper Nile and Warrap get no rows.** The first three were never sampled (the
war of December 2013 was fought there). Warrap was sampled in its towns only, and in wave 1 the
towns of the two neighbouring Dinka states differ from their countryside exactly on the answer
that matters most here: Northern Bahr el Ghazal is 7.2% traditional in town and 24.8% in the
countryside, Eastern Equatoria 2.1% and 13.4%. Warrap's towns (6.3% traditional, 149 heads)
would draw the whole state at a town's rate. Ask 042's recommendation, with Warrap the
builder's call.

**Weights.** `weight` is a household weight ("population weight based on listing scaled to
Census"), constant within each EA; `hhsize` equals the household's roster lines in `hhm`
exactly (asserted), so a person weight is `weight * hhsize`. Wave 1's persons sum to 4.39
million in the six states.

**Two answers.** 33 heads ticked two boxes (22 Christianity and Traditional, 8 Christianity and
Islam, one each with Judaism and Agnostic). Each household is split equally between its
answers. The card lists Christianity first, so `religion1` is card order, not the head's
first choice, and taking it alone would have put all 33 on Christianity.

**Not recorded.** 58 wave-1 heads have no answer, 47 of them in rural Eastern Equatoria
(10% of its rural sample). They are a category of the partition (`NOT_RECORDED`), mapped to
nothing, so they are the not-drawn part of the state.

Usage:
    python sources/ss.py
"""

import io
import itertools
import os
import sys
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

RAW = os.path.join(ROOT, "data", "raw", "ss")
W1_ZIP = os.path.join(RAW, "SSD_2015_HFS-W1_v02_M_STATA8.zip")
W2_ZIP = os.path.join(RAW, "SSD_2016_HFS-W2_v02_M_STATA8.zip")
W1_DIR = "SSD_2015_HFS-W1_v02_M_STATA8/1-CleanOutput/"
W2_DIR = "SSD_2016_HFS-W2_v02_M_STATA8/1-CleanOutput/"
LOOKUP = os.path.join(ROOT, "data", "geo", "ss", "ss_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "ss.csv")

SOURCE_ID = "ss_worldbank_hfsss_w1_2015"

# The card, verbatim from wave 1's `lreligion` value labels (wave 2's `hhh_religion` is the same
# card with "Other (Specify)"). These strings are `source_category` and what taxonomy/ss2015.py keys on.
CATEGORY = {
    1: "Christianity",
    2: "Islam",
    3: "Traditional African Religion",
    4: "Judaism",
    5: "Buddhism",
    6: "Hinduism",
    7: "Agnostic",
    8: "Atheism",
    1000: "Other, please specify",
}
NOT_RECORDED = "Not recorded"
NR_CODE = -1

# Wave 1's `lState`: the six sampled states. Wave 2 adds 81 Warrap, which is not drawn.
STATES_W1 = {82, 83, 84, 91, 92, 93}
WARRAP = 81
N_STATES = 6
NOT_DRAWN = {"SS03": "Jonglei", "SS06": "Unity", "SS07": "Upper Nile", "SS08": "Warrap"}

# Categories the split-half cannot rank at all (too few heads in too few states). lits.stability
# stops unless each is named here with a reason; `compose` draws them at the national share.
UNTESTED = {
    "Agnostic": "one head, as a second answer beside Christianity",
}

# Categories drawn on their own state shares: lits.stability's verdict, asserted in main() on
# every run. Measured 2026-10-03 (p is 1 + draws over 401): Christianity, Traditional African
# Religion, Not recorded and Islam at p 0.0025; Atheism 0.0050 (8 heads, rural Eastern Equatoria
# and Western Bahr el Ghazal); Buddhism 0.0274 (3 heads, all Central Equatoria), both through
# the chi-square and the one-EA veto. Judaism fails (2 heads) and Agnostic cannot be tested
# (1 head); both are drawn at the national share in every state.
CARRIES = None


def _read(zpath, member, labels=False):
    with zipfile.ZipFile(zpath) as z:
        b = z.read(member)
    rd = pd.io.stata.StataReader(io.BytesIO(b))
    df = rd.read(convert_categoricals=False)
    if labels:
        return df, rd.value_labels()
    return df


def check_labels(vals, key, expect_extra=()):
    got = {int(k): v for k, v in vals[key].items() if int(k) < 2_000_000_000}
    want = dict(CATEGORY)
    for k, v in expect_extra:
        want[k] = v
    if got != want:
        raise SystemExit(f"{key} labels changed: {got} against {want}")


def load_wave(zpath, d, wcol, wave):
    """Households with a person weight, and one row per (household, answer)."""
    h, vals = _read(zpath, d + "hhq.dta", labels=True)
    m = _read(zpath, d + "hhm.dta")
    key = ["state", "ea", "hh"]
    if h[key].duplicated().any():
        raise SystemExit(f"wave {wave}: (state, ea, hh) is not unique in hhq")
    n = m.groupby(key).size().rename("roster")
    j = h.set_index(key)[["hhsize"]].join(n)
    if not (j["hhsize"] == j["roster"]).all():
        raise SystemExit(f"wave {wave}: hhsize differs from the roster in "
                         f"{int((j['hhsize'] != j['roster']).sum())} households")
    if h.groupby(["state", "ea"])[wcol].nunique().max() != 1:
        raise SystemExit(f"wave {wave}: {wcol} varies inside an EA; it is not the design weight "
                         "this file assumes")
    if wave == 1:
        check_labels(vals, "lreligion")
        state_lab = {int(k): v for k, v in vals["lState"].items() if int(k) < 2_000_000_000}
    else:
        got = {int(k): v for k, v in vals["hhh_religion"].items() if int(k) < 2_000_000_000}
        if got != {**CATEGORY, 1000: "Other (Specify)"}:
            raise SystemExit(f"wave 2 hhh_religion labels changed: {got}")
        state_lab = {int(k): v for k, v in vals["lState"].items() if int(k) < 2_000_000_000}
    h = h.copy()
    h["pw"] = h[wcol].astype(float) * h["hhsize"].astype(float)
    h["psu"] = h["state"].astype(int) * 1000 + h["ea"].astype(int)
    print(f"  wave {wave}: {len(h):,} households in {h['psu'].nunique()} EAs, "
          f"{h['hhsize'].sum():,} people on the roster (hhsize matches it in every household), "
          f"person weight sums to {h['pw'].sum():,.0f}")

    r1, r2 = h["C_9_hhh_religion1"], h["C_9_hhh_religion2"]
    codes = set(r1.dropna().astype(int)) | set(r2.dropna().astype(int))
    if codes - set(CATEGORY):
        raise SystemExit(f"wave {wave}: religion codes with no label: {sorted(codes - set(CATEGORY))}")
    if (r1.isna() & r2.notna()).any():
        raise SystemExit(f"wave {wave}: a second answer with no first")
    two = r2.notna() & (r2 != r1)
    rows = []
    for i, (a, b, t) in enumerate(zip(r1, r2, two)):
        if pd.isna(a):
            rows.append((i, NR_CODE, 1.0))
        elif t:
            rows.append((i, int(a), 0.5))
            rows.append((i, int(b), 0.5))
        else:
            rows.append((i, int(a), 1.0))
    ix, code, frac = map(np.array, zip(*rows))
    long = h.iloc[ix][["state", "ea", "psu", "pw"] + (["urban"] if "urban" in h else [])].copy()
    long["code"] = code
    long["w"] = long["pw"].to_numpy() * frac
    long["frac"] = frac
    print(f"    {int(two.sum())} heads gave two answers, split equally between them; "
          f"{int(r1.isna().sum())} gave none")
    return long.reset_index(drop=True), state_lab


def fold(s):
    import re
    return re.sub(r"[^a-z]+", "", str(s).casefold())


def decode(df, state_lab, lut):
    """The survey's own state labels to COD-AB pcodes, by name, exactly."""
    by_name = {fold(n): u for n, u in zip(lut["name"], lut["unit"])}
    m = {}
    for code in sorted(set(df["state"].astype(int))):
        u = by_name.get(fold(state_lab[code]))
        if u is None:
            raise SystemExit(f"survey state {code} {state_lab[code]!r} matches no COD-AB state")
        m[code] = u
    if len(set(m.values())) != len(m):
        raise SystemExit(f"two survey states decode to one pcode: {m}")
    return m


def held_out(df, pop, names):
    """Each state's share of the survey's people against the 2025 estimate, over every ordering.

    Printed, not asserted. The decode is the file's own value labels matched to COD-AB's names
    letter for letter, which this cannot improve on; six units allow 719 other orderings, which
    is too few for the check to carry a join (spec §12, São Tomé). Reported so a later reader
    can see the weights and the population base tell roughly the same story.
    """
    s = df.groupby("geo_id")["w"].sum()
    s = s / s.sum()
    p = pop.reindex(s.index) / pop.reindex(s.index).sum()
    a, b = s.to_numpy(), p.to_numpy()
    r = float(np.corrcoef(a, b)[0, 1])
    perms = np.array(list(itertools.permutations(b)))
    rs = np.array([np.corrcoef(a, q)[0, 1] for q in perms])
    beat = int((rs >= r - 1e-12).sum()) - 1
    print(f"\n  held-out (no religion): state share of the survey's people vs the 2025 estimate, "
          f"r = {r:+.3f}; {beat} of 719 other orderings reach it (printed only)")
    for g in s.index:
        print(f"    {names[g]:<26} survey {s[g] * 100:5.1f}%   estimate {p[g] * 100:5.1f}%")


def wave2_witness(w1, w2, names):
    """Wave 2's towns against wave 1's towns, state by state, for the three big answers."""
    print("\n  witness: wave 2 (2016, towns only) against wave 1's urban EAs (2015), person-weighted:")
    u1 = w1[(w1["urban"] == 1) & (w1["code"] != NR_CODE)]
    u2 = w2[w2["code"] != NR_CODE]

    def sh(d):
        t = d.groupby(["geo_id", "code"])["w"].sum().unstack(fill_value=0.0)
        return t.div(t.sum(axis=1), axis=0)
    a, b = sh(u1), sh(u2)
    states = sorted(set(a.index) & set(b.index))
    print(f"    {'state':<26}{'Chr w1':>8}{'w2':>7}{'Isl w1':>8}{'w2':>7}{'Trad w1':>9}{'w2':>7}")
    for g in states:
        f = [a.loc[g].get(c, 0) * 100 for c in (1, 2, 3)] + [b.loc[g].get(c, 0) * 100 for c in (1, 2, 3)]
        print(f"    {names[g]:<26}{f[0]:8.1f}{f[3]:7.1f}{f[1]:8.1f}{f[4]:7.1f}{f[2]:9.1f}{f[5]:7.1f}")
    for c, lab in ((2, "Islam"), (3, "Traditional")):
        x = np.array([a.loc[g].get(c, 0) for g in states])
        y = np.array([b.loc[g].get(c, 0) for g in states])
        print(f"    {lab}: r = {np.corrcoef(x, y)[0, 1]:+.3f} over {len(states)} states' towns, "
              f"mean absolute difference {np.abs(x - y).mean() * 100:.1f} points")


def main():
    import lits

    for p in (W1_ZIP, W2_ZIP, LOOKUP):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing (the zips are Anita's download, ask 042; the lookup is "
                             "sources/ss_geo.py)")
    lut = pd.read_csv(LOOKUP, dtype={"unit": str})
    print("South Sudan: High Frequency South Sudan Survey, World Bank and NBS")
    w1, lab1 = load_wave(W1_ZIP, W1_DIR, "weight", 1)
    w2, lab2 = load_wave(W2_ZIP, W2_DIR, "weight_x", 2)
    if set(w1["state"].astype(int)) != STATES_W1:
        raise SystemExit(f"wave 1 states {sorted(set(w1['state']))}, expected {sorted(STATES_W1)}")
    if set(w2["state"].astype(int)) != STATES_W1 | {WARRAP}:
        raise SystemExit("wave 2 is not the six states plus Warrap")
    m1, m2 = decode(w1, lab1, lut), decode(w2, lab2, lut)
    if any(m2[k] != v for k, v in m1.items()) or m2[WARRAP] != "SS08":
        raise SystemExit(f"the two waves' state labels decode differently: {m1} {m2}")
    w1["geo_id"] = w1["state"].astype(int).map(m1)
    w2["geo_id"] = w2["state"].astype(int).map(m2)
    names = dict(zip(lut["unit"], lut["name"]))
    pop = pd.Series(dict(zip(lut["unit"], lut["pop"].astype("int64"))))
    units = sorted(w1["geo_id"].unique())
    if set(units) & set(NOT_DRAWN):
        raise SystemExit("a not-drawn state has wave-1 respondents")
    print(f"  decoded by name to {', '.join(f'{u} {names[u]}' for u in units)}")

    held_out(w1, pop, names)
    wave2_witness(w1, w2, names)

    # ---- the split-half, on EAs inside states (lits.py's construction) ----
    cat = {**CATEGORY, NR_CODE: NOT_RECORDED}
    df = w1.copy()
    df["code"] = df["code"].map(cat)
    df["PSU_number"] = df["psu"]
    nat = df.groupby("code")["w"].sum() / df["w"].sum()
    print("\n  national (six states), person-weighted:")
    for c, v in nat.sort_values(ascending=False).items():
        print(f"    {v * 100:7.3f}%  {c}  ({int((df['code'] == c).sum())} heads)")
    carries = lits.stability(df, nat, units, unit_col="geo_id", untested=UNTESTED)
    global CARRIES
    expect = ["Christianity", "Traditional African Religion", "Not recorded", "Islam", "Atheism",
              "Buddhism"]
    if sorted(carries) != sorted(expect):
        raise SystemExit(f"the split-half now carries {carries}, this file was written for "
                         f"{expect}; read the table above and edit deliberately")
    CARRIES = carries

    # ---- shares x population: carried at the state's share, the rest flat at the national ----
    by = df.groupby(["geo_id", "code"])["w"].sum().unstack(fill_value=0.0)
    share = by.div(by.sum(axis=1), axis=0)
    tail = [c for c in nat.index if c not in carries]
    tail_total = float(nat[tail].sum())
    carried = share[carries].sum(axis=1)
    print(f"\n  the tail ({', '.join(tail)}) is {tail_total:.2%} nationally and drawn at that in "
          f"every state; as measured it ran {(1 - carried).min():.2%} to {(1 - carried).max():.2%}")
    rows = []
    for u in units:
        p = int(pop[u])
        for c in sorted(nat.index):
            s = (share.loc[u, c] / carried[u] * (1 - tail_total)) if c in carries else nat[c]
            rows.append((u, c, s * p))
    out = pd.DataFrame(rows, columns=["geo_id", "source_category", "count"])
    out["count"] = out["count"].round().astype("int64")
    target = int(pop[units].sum())
    drift = target - int(out["count"].sum())
    if abs(drift) > len(out):
        raise SystemExit(f"rounding drift {drift}")
    if drift:
        out.loc[out["count"].idxmax(), "count"] += drift
    n_by = df.groupby("geo_id")["frac"].sum().round().astype(int)
    out["geo_level"] = "state"
    out["geo_name"] = out["geo_id"].map(names)
    out["basis"] = "self_id"
    out["year"] = "2015"
    out["source_id"] = SOURCE_ID
    out["note"] = [
        f"World Bank and NBS High Frequency South Sudan Survey wave 1 (2015), religion of the "
        f"household head, {int(n_by[g]):,} heads in 50 EAs; "
        + ("state share" if c in carries else "national share, the split-half not passing")
        + ", on the state's 2025 county-based population estimate"
        for g, c in zip(out["geo_id"], out["source_category"])]
    if int(out["count"].sum()) != target:
        raise SystemExit("drawn total is not the six states' population")

    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
            "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} ({len(out)} rows, {target:,} people in {len(units)} states)")
    allpop = int(pop.sum())
    print(f"  not drawn: {', '.join(NOT_DRAWN.values())}, {int(pop[list(NOT_DRAWN)].sum()):,} of "
          f"{allpop:,} ({pop[list(NOT_DRAWN)].sum() / allpop:.2%})")

    show = out.pivot_table(index="geo_id", columns="source_category", values="count", aggfunc="sum")
    show = show.div(show.sum(axis=1), axis=0) * 100
    print(f"\n    {'state':<26}{'heads':>6}{'Chr':>7}{'Trad':>7}{'Isl':>7}{'NR':>6}")
    for g in show.index:
        print(f"    {names[g]:<26}{n_by[g]:>6}{show.loc[g, 'Christianity']:7.1f}"
              f"{show.loc[g, 'Traditional African Religion']:7.1f}{show.loc[g, 'Islam']:7.1f}"
              f"{show.loc[g, NOT_RECORDED]:6.1f}")
    tot = out.groupby("source_category")["count"].sum() / target * 100
    print("  six states as drawn: " + ", ".join(f"{c} {v:.2f}%" for c, v in tot.sort_values(ascending=False).items()))


if __name__ == "__main__":
    main()
