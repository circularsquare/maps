"""South Sudan: the household head's tribe, High Frequency South Sudan Survey, read as language.

    python sources/ss_hfs.py     religiondots' HFSSS zips (read-only, Anita's download, its ask 042)
                                 -> data/normalized/ss.csv        (state x answer, counts)
                                 -> data/normalized/ss_birth.csv  (state x birth county x answer,
                                                                   weights; placement only)

South Sudan has never asked a language in a census (the 2008 Sudan census asked none; neither did
1973-1993). No open survey asks a home language below the nation with all states in it: South
Sudan is not in Afrobarometer, Arab Barometer or WVS; CLEAR Global has no South Sudan layer;
REACH and IOM put none on HDX. The World Bank's 2025 High Frequency Phone Survey does ask "the
primary language spoken in your household" in all ten states, but needs a Microdata Library login
(sources/ss.md section 1). So this is AGENT_BRIEF section 2's ethnicity route, from a survey:
every row `modelled`.

THE QUESTION. Wave 1 (2015, catalog 2778) hhq C.8 "Which tribe does [head] belong to?", one answer
from a card of 67 groups plus "Other (Specify)", whose verbatim is C.8.1. Wave 2 (2016, catalog
2777) asks the same. The answer is the head's; everyone in the household is drawn on it.

STATES. Wave 1 sampled six of the ten former states, towns and countryside: Central, Eastern and
Western Equatoria, Lakes, Northern and Western Bahr el Ghazal. Wave 2 adds Warrap, towns only;
Warrap is drawn from wave 2 (sources/ss.md section 3 says why a town sample will do for
language there and did not for religion). Jonglei, Unity and Upper Nile are in neither wave and
are not drawn.

WEIGHTS. As religiondots' sources/ss.py: `weight` (wave 2 `weight_x`) is a household weight
constant within each EA; a person weight is weight x hhsize (hhsize equals the roster in hhm,
asserted there). Shares per state times the state's 2025 county-based population estimate
(religiondots' ss_lookup.csv `pop`), largest-remainder rounding.

PLACEMENT (ss_birth.csv). The survey has no county of residence, but it has the head's state and
county of birth. Heads born in the state they live in are tabulated by birth county; countries/
ss.py leans each language's dots towards the counties where that state's heads of the group were
born. The counts stay the state's.
"""
import io
import os
import re
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RD = HERE.parent / "religiondots"
RAW = RD / "data" / "raw" / "ss"
W1_ZIP = RAW / "SSD_2015_HFS-W1_v02_M_STATA8.zip"
W2_ZIP = RAW / "SSD_2016_HFS-W2_v02_M_STATA8.zip"
W1_DIR = "SSD_2015_HFS-W1_v02_M_STATA8/1-CleanOutput/"
W2_DIR = "SSD_2016_HFS-W2_v02_M_STATA8/1-CleanOutput/"
LOOKUP = RD / "data" / "geo" / "ss" / "ss_lookup.csv"
COUNTIES = RD / "data" / "geo" / "ss" / "ss_counties.csv"
OUT = HERE / "data" / "normalized" / "ss.csv"
OUT_BIRTH = HERE / "data" / "normalized" / "ss_birth.csv"

SOURCE_ID = "ss_worldbank_hfsss_w1_w2_tribe"
WARRAP = "SS08"
SIX = {"SS01", "SS02", "SS04", "SS05", "SS09", "SS10"}
NOT_DRAWN = {"SS03": "Jonglei", "SS06": "Unity", "SS07": "Upper Nile"}
NOT_STATED = "Not stated"
OTHER = "Other (Specify)"

# Survey birth-county spellings that differ from COD-AB's (after dropping " County" and folding).
COUNTY_ALIAS = {"rumbekcenter": "rumbekcentre", "lafonlopa": "lafon", "raga": "raja",
                "meluth": "melut", "panriang": "pariang"}


def fold(s):
    return re.sub(r"[^a-z]+", "", str(s).casefold())


def say(ok, msg):
    print(("  ok  " if ok else "  FAIL") + "  " + msg)
    if not ok:
        raise SystemExit("check failed: " + msg)


def _read(zpath, member):
    with zipfile.ZipFile(zpath) as z:
        b = z.read(member)
    rd = pd.io.stata.StataReader(io.BytesIO(b))
    df = rd.read(convert_categoricals=False)
    rd2 = pd.io.stata.StataReader(io.BytesIO(b))
    rd2.read(convert_categoricals=False)
    return df, rd2.value_labels(), rd2.variable_labels()


def lab(vals, var_labels_key):
    return {int(k): v for k, v in vals[var_labels_key].items() if int(k) < 2_000_000_000}


def load(zpath, d, wcol, wave, lut, counties):
    h, vals, vlab = _read(zpath, d + "hhq.dta")
    m, _, _ = _read(zpath, d + "hhm.dta")
    key = ["state", "ea", "hh"]
    say(not h[key].duplicated().any(), f"wave {wave}: (state, ea, hh) unique in hhq, {len(h):,} households")
    n = m.groupby(key).size().rename("roster")
    j = h.set_index(key)[["hhsize"]].join(n)
    say(bool((j["hhsize"] == j["roster"]).all()), f"wave {wave}: hhsize equals the hhm roster in every household")
    say(h.groupby(["state", "ea"])[wcol].nunique().max() == 1, f"wave {wave}: {wcol} constant within each EA")

    # value labels, found through each variable's label-set name
    with zipfile.ZipFile(zpath) as z:
        rd = pd.io.stata.StataReader(io.BytesIO(z.read(d + "hhq.dta")))
        rd.read(convert_categoricals=False)
        lblname = dict(zip(rd._varlist, rd._lbllist))
    tribe_v = "C_8_hhh_tribe"
    spec_v = "C_8_1_hhh_tribe_spec"
    bst_v, bco_v = "C_2_1_hhh_birthplace_ssd", "C_2_2_countyborn"
    say("tribe" in vlab[tribe_v].lower(), f"wave {wave}: {tribe_v} is {vlab[tribe_v]!r}")
    tl = lab(vals, lblname[tribe_v])
    sl = lab(vals, lblname["state"])
    bsl = lab(vals, lblname[bst_v])
    bcl = lab(vals, lblname[bco_v])

    by_name = {fold(nm): u for nm, u in zip(lut["name"], lut["unit"])}
    h = h.copy()
    h["geo_id"] = h["state"].astype(int).map(lambda c: by_name.get(fold(sl[c])))
    say(h["geo_id"].notna().all(), f"wave {wave}: every state label matches a COD-AB state by name")
    t = h[tribe_v]
    say(set(t.dropna().astype(int)) <= set(tl), f"wave {wave}: every tribe code has a label")
    cat = t.map(lambda c: NOT_STATED if pd.isna(c) else tl[int(c)])
    spec = h[spec_v].fillna("").astype(str).str.strip()
    is_other = cat == OTHER
    say(bool((spec[is_other] != "").all()), f"wave {wave}: every 'Other (Specify)' has a verbatim "
        f"({int(is_other.sum())})")
    say(bool((spec[~is_other] == "").all()), f"wave {wave}: no verbatim beside a card answer")
    cat = cat.where(~is_other, "Other: " + spec)
    h["source_category"] = cat
    h["pw"] = h[wcol].astype(float) * h["hhsize"].astype(float)

    # birth state and county; county matched within the birth state, exactly, by folded name
    cn = counties.copy()
    cn["cname"] = cn["name"].str.split(",").str[0].map(fold)
    cmap = {(u, c): k for u, c, k in zip(cn["unit"], cn["cname"], cn["county"])}
    bstate = h[bst_v].map(lambda c: None if pd.isna(c) else by_name.get(fold(bsl[int(c)])))
    named_b = h[bst_v].dropna().astype(int).map(bsl)
    unmatched_states = sorted({s for s in named_b if by_name.get(fold(s)) is None})
    # Abyei has no state polygon in COD-AB's ten; a birth there is simply out of state
    say(set(unmatched_states) <= {"Abyei"}, f"wave {wave}: birth states match COD-AB except Abyei "
        f"({unmatched_states})")

    def county(row):
        c, u = row[bco_v], row["_bst"]
        if pd.isna(c) or u is None:
            return None
        f = fold(re.sub(r"\s*County\s*$", "", bcl[int(c)]))
        f = COUNTY_ALIAS.get(f, f)
        return cmap.get((u, f), "MISS:" + bcl[int(c)] + "|" + u)
    h["_bst"] = bstate
    h["bcounty"] = h.apply(county, axis=1)
    miss = sorted({x for x in h["bcounty"].dropna() if x.startswith("MISS:")})
    say(not miss, f"wave {wave}: every birth county matches a COD-AB county of its birth state {miss[:5]}")
    h["born_here"] = h["_bst"] == h["geo_id"]
    return h


def main():
    for p in (W1_ZIP, W2_ZIP, LOOKUP, COUNTIES):
        if not p.exists():
            raise SystemExit(f"{p} missing (the zips are Anita's download for religiondots' ask 042)")
    lut = pd.read_csv(LOOKUP, dtype={"unit": str})
    counties = pd.read_csv(COUNTIES, dtype=str)
    say(len(lut) == 10 and len(counties) == 78, "10 states and 78 counties (Abyei apart) in religiondots' lookups")
    names = dict(zip(lut["unit"], lut["name"]))
    pop = pd.Series(dict(zip(lut["unit"], lut["pop"].astype("int64"))))

    print("wave 1 (2015)")
    w1 = load(W1_ZIP, W1_DIR, "weight", 1, lut, counties)
    say(set(w1["geo_id"]) == SIX, f"wave 1 is the six states {sorted(SIX)}")
    print("wave 2 (2016)")
    w2 = load(W2_ZIP, W2_DIR, "weight_x", 2, lut, counties)
    say(set(w2["geo_id"]) == SIX | {WARRAP}, "wave 2 is the six states plus Warrap")

    # ---- witness: wave 2's towns against wave 1's towns, per state ----
    u1 = w1[w1["urban"].astype(int) == 1]
    def sh(d):
        t = d.groupby(["geo_id", "source_category"])["pw"].sum().unstack(fill_value=0.0)
        return t.div(t.sum(axis=1), axis=0)
    a, b = sh(u1), sh(w2[w2["geo_id"] != WARRAP])
    cats = sorted(set(a.columns) | set(b.columns))
    a, b = a.reindex(columns=cats, fill_value=0.0), b.reindex(columns=cats, fill_value=0.0)
    print("\n  witness: each state's top answer in wave 2's towns (2016) and wave 1's towns (2015)")
    diffs = []
    for g in sorted(a.index):
        top = b.loc[g].idxmax()
        diffs.append(float((a.loc[g] - b.loc[g]).abs().sum() / 2))
        print(f"    {names[g]:<26} {top:<28} w2 {b.loc[g, top]:.3f}  w1 {a.loc[g, top]:.3f}   "
              f"dissimilarity {diffs[-1]:.3f}")
    x, y = a.to_numpy().ravel(), b.to_numpy().ravel()
    print(f"    every (state, answer) cell: r = {np.corrcoef(x, y)[0, 1]:+.3f} over {x.size} cells")

    # ---- town and country in wave 1's Dinka states: does a town sample stand for Warrap? ----
    print("\n  Dinka share of persons, wave 1, towns against countryside:")
    for g in ("SS05", "SS04"):
        d = w1[w1["geo_id"] == g]
        for u, lbl in ((1, "towns"), (0, "countryside")):
            s = d[d["urban"].astype(int) == u]
            share = s.loc[s["source_category"].str.startswith("Dinka"), "pw"].sum() / s["pw"].sum()
            print(f"    {names[g]:<26}{lbl:<12} {share:.3f}  ({len(s)} heads)")
    ww = w2[w2["geo_id"] == WARRAP]
    print(f"    Warrap (wave 2, towns)               "
          f"{ww.loc[ww['source_category'].str.startswith('Dinka'), 'pw'].sum() / ww['pw'].sum():.3f}"
          f"  ({len(ww)} heads)")

    # ---- the drawn table: wave 1 for six states, wave 2 for Warrap ----
    use = pd.concat([w1.assign(wave=1), ww.assign(wave=2)], ignore_index=True)
    rows = []
    for g, d in use.groupby("geo_id"):
        s = d.groupby("source_category")["pw"].sum()
        heads = d.groupby("source_category").size()
        share = s / s.sum()
        exact = share * int(pop[g])
        cnt = np.floor(exact).astype("int64")
        rem = int(pop[g]) - int(cnt.sum())
        order = (exact - cnt).sort_values(ascending=False).index[:rem]
        cnt[order] += 1
        say(int(cnt.sum()) == int(pop[g]), f"{names[g]}: {len(d)} heads, sums to its 2025 estimate "
            f"{int(pop[g]):,}")
        wave = "wave 2 (2016, towns only)" if g == WARRAP else "wave 1 (2015)"
        for c in s.index:
            rows.append(dict(geo_id=g, geo_level="state", geo_name=names[g], source_category=c,
                             count=int(cnt[c]), tier="modelled", heads=int(heads[c]),
                             source_id=SOURCE_ID, year="2016" if g == WARRAP else "2015",
                             note=f"HFSSS {wave}, tribe of the household head, share "
                                  f"{share[c]:.5f} of {len(d)} heads, on the 2025 estimate"))
    out = pd.DataFrame(rows)
    drawn = SIX | {WARRAP}
    say(set(out["geo_id"]) == drawn, "seven states drawn")
    say(int(out["count"].sum()) == int(pop[list(drawn)].sum()),
        f"drawn total {int(out['count'].sum()):,} = the seven states' 2025 estimate")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(HERE)}: {len(out)} rows, {out['source_category'].nunique()} answers")
    allpop = int(pop.sum())
    nd = int(pop[list(NOT_DRAWN)].sum())
    print(f"  not drawn: {', '.join(NOT_DRAWN.values())}, {nd:,} of {allpop:,} ({nd / allpop:.2%})")

    # ---- placement table: heads born in their own state, by birth county ----
    b = use[use["born_here"] & use["bcounty"].notna()]
    print(f"  placement: {len(b):,} of {len(use):,} heads were born in their state of residence "
          f"with a county named ({len(b) / len(use):.1%})")
    bt = (b.groupby(["geo_id", "bcounty", "source_category"])["pw"].sum().rename("w").reset_index()
          .rename(columns={"bcounty": "county"}))
    bt["heads"] = b.groupby(["geo_id", "bcounty", "source_category"]).size().to_numpy()
    for g, d in bt.groupby("geo_id"):
        allc = set(counties.loc[counties["unit"] == g, "county"])
        say(set(d["county"]) <= allc, f"{names[g]}: birth counties are its own "
            f"({d['county'].nunique()} of {len(allc)} named)")
    bt.to_csv(OUT_BIRTH, index=False, encoding="utf-8")
    print(f"wrote {OUT_BIRTH.relative_to(HERE)}: {len(bt)} rows")

    tot = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print("\n  seven states as drawn, top 25:")
    for c, v in tot.head(25).items():
        print(f"    {v / tot.sum() * 100:6.2f}%  {v:>10,}  {c}")


if __name__ == "__main__":
    main()
