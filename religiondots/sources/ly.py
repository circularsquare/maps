"""Libya: religion from the pooled Arab Barometer, one national share, on BSC's 2020 estimate.

Reads data/raw/arabbarometer/*.sav and data/geo/ly/ly_lookup.csv; writes data/normalized/ly.csv.
`sources/ly.md` is the record in prose; `sources.md` §maghreb-2026-09-16 the scouting;
`ask/RULINGS.md` 2026-09-15 and 2026-09-16 the rulings.

## WHAT IS DRAWN

No Libyan census has asked religion, and none has been held since 2006. Arab Barometer waves V to
VII ask about 7,200 Libyan citizens. One thing is measured and drawn: **the national non-Muslim
share, and what the non-Muslims said** (Christian, no religion), applied to every district's
Libyans in the Bureau of Statistics and Census's 2020 estimate. Seven answers cannot place
anything, and the two tests that could are printed and asserted to fail:

  * **no district stands apart** (`standouts`: each district with two or more non-Muslims against
    the rest, in both halves of the waves that carry a usable district, Bonferroni);
  * **towns do not** (`urban_test`, waves V and VII, which record the stratum). Morocco's bar also
    asks for the Afrobarometer, which has never surveyed Libya, so it cannot pass here.

## THE POOL IS WAVES V TO VII

`ab.card` reads each wave's `Q1012` labels. Wave III's card has no box for having no religion, and
no Libyan of its 1,247 answered anything but Muslim (12 said they did not know); it is out. V offers
`Atheist`, VI-1 to VII `No religion`, merged as each card's one box for having none. Libya is not
in wave VIII.

## WAVE VII'S DISTRICT LABELS DO NOT FIT THE POPULATION

`allocation` compares each wave's weighted respondents per district with BSC's 2020 estimate. V,
VI-1, VI-2 and VI-3 rank with it at Spearman +0.99 or better. **VII reads +0.85**, with Ajdabiya
at 3.7 times its population share, Ghat 3.5, Jafara 2.3, and Benghazi 0.32, Murzuq 0.29, Tripoli
0.41, all with mean weights near 1; swapping those three pairs halves the misfit and leaves
Misrata at 0.53 and Wadi al Shati at 1.75, so it is not a clean swap of labels. The wave's
religion answers and weights are used for the national level and mix, and its district labels
for nothing: the held-out, quota and standout tests read V and VI only (`GEO_WAVES`). VII's codes
11001-11022 also mean different districts from VI's 11001-11023, so neither is decoded on code.

Usage:
    python sources/ly.py            rebuild data/normalized/ly.csv
"""

import os
import re
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd
from scipy import stats

import arabbarometer as ab
from afrobarometer import round_within_rows
from ma import stratified_p, wshare          # generic; Morocco's copies, imported not copied
from tn import contradictions                # generic over sect_m / sect_c; Tunisia's copy

LOOKUP = os.path.join(ROOT, "data", "geo", "ly", "ly_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "ly.csv")

COUNTRY = "Libya"
WAVES = ["V", "VI-1", "VI-2", "VI-3", "VII"]
GEO_WAVES = ["V", "VI-1", "VI-2", "VI-3"]
YEARS = "2018-2022"
SOURCE_ID = "ly_arabbarometer_2018_2022"

OMIT = {"III": "the card has no box for having no religion (sources/ly.md §3); and no Libyan of "
               "1,247 answered anything but Muslim, 12 saying they did not know"}
RECODE = {
    "refused": "Refused to answer",                         # wave V
    "Atheist": "No religion",                               # wave V: the card's one box for none
}
DROPPED = {"Refused to answer": "a refusal is not a religion"}
CATS = ["Muslim", "Christian", "No religion"]
NON_MUSLIM = ["Christian", "No religion"]

# Every Q1 label in waves V to VII, folded by `key`, -> COD p-code (= BSC's code).
LABELS = {
    "LY0101": ["derna"], "LY0102": ["almarj", "al marj"], "LY0103": ["benghazi"],
    "LY0104": ["tobruq", "tobruk"], "LY0105": ["al wahat", "ejdabia"],
    "LY0106": ["al jabal al akhdar", "al gabal al akhdar"], "LY0107": ["al kufra", "alkufra"],
    "LY0208": ["sirt"], "LY0209": ["nalut"], "LY0210": ["al murqub", "al mergheb", "almargeb"],
    "LY0211": ["tripoli"], "LY0212": ["jafara", "aljfara"],
    "LY0213": ["al zawia", "zawia", "azzawya"], "LY0214": ["misrata"],
    "LY0215": ["nuqat al khams", "zwara"], "LY0216": ["al jabal al gharbi", "al gabal al gharbi"],
    "LY0317": ["al jofra", "aljufra"], "LY0318": ["wadi alshati", "wadi shati", "wadi ashshati"],
    "LY0319": ["sebha", "sabha"], "LY0320": ["ubari", "wadi al haya"],
    "LY0321": ["ghat"], "LY0322": ["murzuq", "murzuk"],
}
NORM = {s: pc for pc, ss in LABELS.items() for s in ss}
BOGUS = {"don t know", "refused"}      # VI-1 to VI-3, codes 99998 and 99999; must all be Muslim
V_EMPTY = {"LY0317"}                   # Jufra: no respondent in wave V

ALLOC_BAR = 0.95                       # weighted respondents per district against the estimate
VII_ALLOC = (0.80, 0.90)               # VII's measured +0.853, pinned so a re-release shows
HALVES = [["V", "VI-1"], ["VI-2", "VI-3"]]
STANDOUTS = set()
URBAN_BAR = 0.05                       # Morocco's, pre-registered in sources/ma.py
NONMUSLIM_BAND = (0.0002, 0.005)

# Pew Research Center, Religious Composition 2010-2020, percentages file, Libya 2020
# (data/raw/estimates/pew.zip). Printed beside the survey's level, never used to set it. Pew
# counts everyone in the country; this map draws Libyans.
PEW_2020 = {"Muslims": 98.994522, "Christians": 0.524618, "Buddhists": 0.258001,
            "Hindus": 0.089998, "Other_religions": 0.081557, "Religiously_unaffiliated": 0.049394,
            "Jews": 0.001906}

# note_public's survey figures, measured 2026-09-15 and asserted in main.
NOTE = dict(pooled=7196, named=7191, nonmuslim=7, christian=6, no_religion=1, ibadi=5,
            share_pct=0.102, christian_pct=0.084)


def key(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z]", " ", s)).strip()


def decode(df):
    """COD p-code per respondent, by name in every wave; each wave's code-to-label table 1:1."""
    g = df.copy()
    g["k"] = g["geo_raw"].map(key)
    for w, x in g.groupby("wave"):
        pairs = x.groupby("geo_code")["k"].nunique()
        labels = x.groupby("k")["geo_code"].nunique()
        if (pairs > 1).any() or (labels > 1).any():
            raise SystemExit(f"wave {w}: a Q1 code carries two labels or a label two codes")
    bogus = g["k"].isin(BOGUS)
    if (g.loc[bogus, "category"] != "Muslim").any():
        raise SystemExit("a respondent with no district is not Muslim")
    print(f"  {int(bogus.sum())} respondents gave no district (all Muslim); they stay in the "
          "national level and leave the geography")
    g["geo_id"] = g["k"].map(NORM)
    bad = sorted(set(g.loc[g["geo_id"].isna() & ~bogus, "geo_raw"]))
    if bad:
        raise SystemExit(f"Q1 labels with no district: {bad}")
    vi = g[g["wave"].isin(["VI-1", "VI-2", "VI-3"]) & ~bogus]
    both = vi.groupby("wave")["k"].apply(lambda s: {"al wahat", "ejdabia"} <= set(s))
    if not both.all():
        raise SystemExit("wave VI no longer labels Al-Wahat and Ejdabia apart")
    print("  wave VI labels Al-Wahat and Ejdabia apart; both are BSC's Ajdabiya and the Oases "
          "(LY0105)")
    return g, bogus


def allocation(g, pop):
    share = pop / pop.sum()
    print("\n  weighted respondents per district against BSC 2020 (Libyans), by wave:")
    got = {}
    for w in WAVES:
        x = g[g["wave"] == w]
        ws = x.groupby("geo_id")["w"].sum().reindex(pop.index, fill_value=0.0)
        rho = stats.spearmanr(ws, pop).statistic
        rel = (ws / ws.sum()) / share
        empty = set(rel.index[rel == 0])
        got[w] = rho
        print(f"    {w:<5} Spearman {rho:+.3f}; lowest {rel[rel > 0].idxmin()} "
              f"{rel[rel > 0].min():.2f}, highest {rel.idxmax()} {rel.max():.2f}"
              + (f"; no respondent in {sorted(empty)}" if empty else ""))
        if empty != (V_EMPTY if w == "V" else set()):
            raise SystemExit(f"wave {w}: districts with no respondent {sorted(empty)}")
    for w in GEO_WAVES:
        if got[w] < ALLOC_BAR:
            raise SystemExit(f"wave {w}'s districts no longer fit the estimate ({got[w]:+.3f})")
    if not VII_ALLOC[0] <= got["VII"] <= VII_ALLOC[1]:
        raise SystemExit(f"wave VII's allocation reads {got['VII']:+.3f}, outside the pinned "
                         f"{VII_ALLOC}; read sources/ly.md §3 before using its labels")


def standouts(g):
    cand = g.groupby("geo_id")["nonm"].sum()
    cand = cand[cand >= 2]
    found = set()
    if len(cand):
        bar = 0.05 / len(cand)
        print(f"\n  standouts: {len(cand)} districts with two or more non-Muslims, each against the "
              f"rest in both halves ({' | '.join(','.join(h) for h in HALVES)}), bar P < {bar:.4f}:")
        for n in cand.index:
            ps = [stratified_p(d, d["geo_id"] == n) for d in (g[g["wave"].isin(h)] for h in HALVES)]
            print(f"    {n}  " + "   ".join(f"{o} vs {e:.2f} P={p:.3f}" for o, e, p in ps))
            if all(p < bar for _o, _e, p in ps):
                found.add(n)
    else:
        print("\n  standouts: no district holds two non-Muslims")
    if found != STANDOUTS:
        raise SystemExit(f"districts standing apart: {sorted(found)}, not {sorted(STANDOUTS)}")


def urban_test(df):
    d = df[df["urban"].notna()].copy()
    d["u"] = d["urban"].str.lower().eq("urban")
    o, e, p = stratified_p(d, d["u"])
    print(f"\n  urban excess, exact within wave ({','.join(sorted(set(d['wave'])))}): {o} urban of "
          f"{int(d['nonm'].sum())} non-Muslims against {e:.2f} expected, P = {p:.3f}; the "
          "Afrobarometer has no Libyan round, so Morocco's two-survey bar cannot be met")
    if p < URBAN_BAR:
        raise SystemExit("the Arab Barometer alone now shows an urban excess; read sources/ly.md §4")


def main():
    ab.card(WAVES, OMIT)
    print("\n=== Arab Barometer, Libya ===")
    df = ab.load(COUNTRY, expect_waves=WAVES, waves=WAVES, omit=OMIT, recode=RECODE,
                 extra={"sect_m": ("Q1012A_MUSLIM", "q1012a"), "sect_c": ("Q1012A_CHRISTIAN",),
                        "urban": ("q13",)},
                 raw={"psu": ("psu",)})
    pooled = len(df)
    print(pd.crosstab(df["category"], df["wave"]).to_string())
    for cat, why in DROPPED.items():
        print(f"  dropping {int((df['category'] == cat).sum())} {cat!r}: {why}")
        df = df[df["category"] != cat]
    odd = sorted(set(df["category"]) - set(CATS))
    if odd:
        raise SystemExit(f"answers with no category here: {odd}; read them before adding one")
    named = len(df)
    contradictions(df)
    df["nonm"] = df["category"] != "Muslim"

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if len(lut) != 22:
        raise SystemExit("ly_lookup.csv is not the 22 districts")
    pop = lut.set_index("geo_id")["pop"]

    g, bogus = decode(df)
    allocation(g[~bogus], pop)
    geo = g[~bogus & g["wave"].isin(GEO_WAVES)]
    ab.held_out(geo, pop, COUNTRY, pop_source="BSC 2020, Libyans")
    ab.assert_not_quota(geo, COUNTRY)
    standouts(geo)
    urban_test(df)

    print("\n  every non-Muslim answer:")
    for _i, r in g[g["nonm"]].iterrows():
        print(f"    {r['wave']:<5} {r['category']:<12} {r['geo_raw']:<22} w={r['w']:.3f} "
              f"urban={r['urban']} follow-up={r['sect_c'] if isinstance(r['sect_c'], str) else r['sect_m']}")

    L = wshare(df, df["nonm"])
    print(f"\n  weighted non-Muslim share, waves V-VII: {L:.4%}; by wave "
          + ", ".join(f"{w} {wshare(x, x['nonm']):.3%}" for w, x in df.groupby('wave', sort=False)))
    if not NONMUSLIM_BAND[0] <= L <= NONMUSLIM_BAND[1]:
        raise SystemExit(f"{L:.4%} is outside {NONMUSLIM_BAND}")
    nm = df[df["nonm"]]
    comp = (nm.groupby("category")["w"].sum() / nm["w"].sum()).reindex(NON_MUSLIM, fill_value=0.0)
    print("    non-Muslim mix (weighted): " + ", ".join(
        f"{c} {v:.1%} (n={int((nm['category'] == c).sum())})" for c, v in comp.items()))
    print("  outside level check, Pew Research Center 2020 (everyone in Libya): "
          + ", ".join(f"{k} {v:.3f}%" for k, v in PEW_2020.items())
          + f"; the survey reads Christian {100 * comp['Christian'] * L:.3f}% and no religion "
          f"{100 * comp['No religion'] * L:.3f}% of Libyans")

    sect = df["sect_m"].map(lambda v: ab.fold(v) if isinstance(v, str) else None)
    ibadi = df[sect.isin(["ibadi", "ibadhi", "mozabite"])]
    print(f"\n  witness, not drawn: Ibadi or Mozabite on the follow-up, {len(ibadi)} in the pool: "
          f"{ibadi.groupby(['wave', 'geo_raw']).size().to_dict()}")
    v_sect = df.loc[df["wave"] == "V", "sect_m"].value_counts().to_dict()
    print(f"  wave V's sect item (interviewer-logged, not used): {v_sect}")

    got = dict(pooled=pooled, named=named, nonmuslim=len(nm),
               christian=int((nm["category"] == "Christian").sum()),
               no_religion=int((nm["category"] == "No religion").sum()),
               ibadi=len(ibadi), share_pct=round(100 * L, 3),
               christian_pct=round(100 * comp["Christian"] * L, 3))
    print(f"\n  note_public's survey figures: {got}")
    if NOTE and got != NOTE:
        raise SystemExit(f"note_public's figures are {NOTE}; the pool gives {got}")

    names = dict(zip(lut["geo_id"], lut["name"]))
    m = pd.DataFrame({"Muslim": pop * (1 - L), **{c: pop * L * comp[c] for c in NON_MUSLIM}})[CATS]
    counts = round_within_rows(m)
    if not (counts.sum(axis=1) == pop.reindex(counts.index)).all():
        raise SystemExit("a district's rounded counts do not sum to its estimate")
    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_id", "source_category", "count"]
    n_by = g[~bogus].groupby("geo_id").size()
    out["geo_level"] = "district"
    out["geo_name"] = out["geo_id"].map(names)
    out["basis"] = "self_id"
    out["year"] = YEARS
    out["source_id"] = SOURCE_ID
    out["note"] = out["geo_id"].map(
        lambda i: f"Arab Barometer waves V to VII pooled, n={int(n_by.get(i, 0))} labelled this "
                  "district; the national non-Muslim share and mix, on BSC's 2020 estimate of "
                  "Libyans")
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
            "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    total = int(out["count"].sum())
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} Libyans, 22 districts)")
    drawn = out.groupby("source_category")["count"].sum()
    for c in CATS:
        print(f"    {drawn[c] / total:9.4%}  {c}  ({int(drawn[c]):,})")


if __name__ == "__main__":
    main()
