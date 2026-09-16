"""Tunisia: religion from the pooled Arab Barometer, one national share, on the 2024 census.

Reads data/raw/arabbarometer/*.sav, data/raw/afrobarometer/ (a witness) and
data/geo/tn/tn_lookup.csv; writes data/normalized/tn.csv. `sources/tn.md` is the record in prose;
`sources.md` §maghreb-2026-09-16 the scouting; `ask/RULINGS.md` 2026-09-15 and 2026-09-16 the
rulings.

## WHAT IS DRAWN

No Tunisian census asks religion (the 2014 form was read, `sources.md` §11aq). Arab Barometer
waves V to VIII ask about 10,400 adults. One thing is measured and drawn: **the national
non-Muslim share, and what the non-Muslims said** (Christian, no religion, other), applied to
every governorate's 2024 census count. Nothing finer holds:

  * **no governorate stands apart** (`standouts`: each governorate with two or more non-Muslims
    against the rest, in both halves of the waves, Bonferroni);
  * **Greater Tunis does not** (`greater_tunis`: the queue row's "slight lean to Tunis"; Tunis,
    Ariana, Ben Arous and Manouba read 0.61% against 0.62% for the rest);
  * **towns do not** (`urban_test`: Morocco's bar, P < 0.05 in the Arab Barometer waves that
    record the stratum AND in the Afrobarometer; Tunisia reads P 0.41 and 0.12).

Each test is asserted to come out as it did, so a re-release that changes one stops the build.
The probe that measured them ran before this file was written; Morocco's urban bar was
pre-registered there and is used here unchanged.

## THE POOL IS WAVES V TO VIII

`card()` reads each wave's `Q1012` labels. Waves II, III and IV offer no box for having no
religion, which is 28 of the 65 non-Muslim answers where offered; III and IV also record no
non-Muslim at all among 2,399 Tunisians. V offers `Atheist`, VI-1 to VIII `No religion`; they are
merged as each card's one box for having none.

## WAVE VI LABELS TWO GOVERNORATES `Jendouba`

VI-1, VI-2 and VI-3 code Tunisia's governorates 21001 to 21024 in their own order and label both
21009 and 21010 `Jendouba`; no `Kef` label exists. A name join merges Le Kef into Jendouba with
every total intact (Yemen's wave III trap, `playbooks/arabbarometer.md`). VI is decoded on the
code (`VI_CODES`), with the names as a witness everywhere else, and 21009 is Le Kef because of
sample size: 23, 23 and 28 respondents against 37, 37 and 44, where the other waves interview
56 in Le Kef for every 88 in Jendouba (`decode`, asserted against the swap).

## WAVE VII'S `Other`

14 answers, 10 of them in Le Kef, Siliana and Sousse, whose PSUs run consecutively (139 to 164);
none of those three governorates has a non-Muslim in waves V to VI-3, and the files carry no
interviewer or team column to test a fieldwork habit. They stay in the pool as recorded: they are
not placed (no governorate stands apart), and without them the national share is about 0.52%
rather than 0.62% (`vii_cluster` prints both).

Usage:
    python sources/tn.py            rebuild data/normalized/tn.csv
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
import afrobarometer as afro
from afrobarometer import round_within_rows
from ma import stratified_p, wshare          # generic; Morocco's copies, imported not copied
from stability import CELL_CAP

LOOKUP = os.path.join(ROOT, "data", "geo", "tn", "tn_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "tn.csv")

COUNTRY = "Tunisia"
WAVES = ["V", "VI-1", "VI-2", "VI-3", "VII", "VIII"]
YEARS = "2018-2024"
SOURCE_ID = "tn_arabbarometer_2018_2024"

_NO_BOX = ("the card has no box for having no religion, which is 28 of the 65 non-Muslim answers "
           "in the waves that offer one; pooling it would mix two questionnaires (sources/tn.md §3)")
OMIT = {"II": _NO_BOX,
        "III": _NO_BOX + "; and no Tunisian of 1,199 answered anything but Muslim",
        "IV": _NO_BOX + "; and no Tunisian of 1,200 answered anything but Muslim"}
RECODE = {
    "refused": "Refused to answer",                         # wave V
    "Refused": "Refused to answer",                         # wave VI-1
    "other": "Other",                                       # wave V
    "Something else: SPECIFY_______": "Other",              # wave VI-3's card has no plain Other
    "Atheist": "No religion",                               # wave V: the card's one box for none
}
DROPPED = {"Refused to answer": "a refusal is not a religion"}
CATS = ["Muslim", "Christian", "No religion", "Other"]
NON_MUSLIM = ["Christian", "No religion", "Other"]
SECT_ISLAM = {"sunni", "shia", "just a muslim", "shafi'i", "hanbali", "maliki", "malki", "hanafi",
              "ibadi", "sufi", "ja'fari", "ahmadiyya"}
SECT_CHRISTIAN = {"catholic", "orthodox", "coptic", "protestant", "just a christian",
                  "evangelical"}

# INS governorate code -> every Q1 spelling in waves V to VIII, folded by `key`.
SPELLINGS = {11: ["tunis"], 12: ["ariana"], 13: ["ben arous"], 14: ["manouba"], 15: ["nabeul"],
             16: ["zaghouan"], 17: ["bizerte"], 21: ["beja"], 22: ["jendouba"], 23: ["kef"],
             24: ["siliana"], 31: ["sousse"], 32: ["monastir"], 33: ["mahdia"], 34: ["sfax"],
             41: ["kairouan"], 42: ["kasserine"], 43: ["sidi bouzid"], 51: ["gabes"],
             52: ["medenine"], 53: ["tatouine", "tataouine"], 61: ["gafsa"], 62: ["tozeur"],
             63: ["kebili"]}
NORM = {s: n for n, ss in SPELLINGS.items() for s in ss}
BOGUS = {"refused"}                    # VI-1 and VI-2, code 99999; must all be Muslim

# Waves V, VII and VIII: Q1 - 210000 is the position in INS's official order.
OFFICIAL_ORDER = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 31, 32, 33, 34, 41, 42, 43,
                  51, 52, 53, 61, 62, 63]
NAMED_WAVES = {"V": 210000, "VII": 210000, "VIII": 210000}
# Waves VI-1, VI-2, VI-3: Q1 - 21000 -> INS code. Decoded on the code, not the label.
VI_BASE = 21000
VI_CODES = {1: 11, 2: 12, 3: 13, 4: 14, 5: 17, 6: 15, 7: 21, 8: 16, 9: 23, 10: 22, 11: 24,
            12: 31, 13: 32, 14: 33, 15: 41, 16: 34, 17: 43, 18: 42, 19: 61, 20: 62, 21: 53,
            22: 52, 23: 51, 24: 63}
VI_MISLABELLED = {9: "jendouba"}       # code 21009 is labelled Jendouba and is Le Kef

HALVES = [["V", "VI-1", "VI-2", "VI-3"], ["VII", "VIII"]]
STANDOUTS = set()                      # governorates drawn apart; asserted
GREATER_TUNIS = {11, 12, 13, 14}
GREATER_TUNIS_EXPECTED = False
URBAN_BAR = 0.05                       # Morocco's, pre-registered in sources/ma.py
URBAN_EXPECTED = False
NONMUSLIM_BAND = (0.001, 0.012)
VII_CLUSTER = {23, 24, 31}             # Le Kef, Siliana, Sousse
VII_CLUSTER_OTHER = (10, 14)           # wave VII Other answers there, of all; asserted

AFRO_ROUNDS = [5, 6, 7, 8, 9]
AFRO_MUSLIM = re.compile(r"(?i)muslim|sunni|shia|isma|qadiri|tijani")
AFRO_NON_MUSLIM = {                    # every non-Muslim label, so a new one stops
    "Agnostic", "Agnostic (Do not know if there is a God)", "Atheist",
    "Atheist (Do not believe in a God)", "Bahai",
    "Christian only (i.e., respondents says only “Christian”, without identifying a specific "
    "sub-group)", "Hindu", "Jewish", "None", "Other", "Seventh Day Adventist",
    "Traditional / ethnic religion"}
AFRO_GREATER_TUNIS = {"tunis", "ariana", "ben arous", "manouba", "great tunis"}

# Pew Research Center, Religious Composition 2010-2020, percentages file, Tunisia 2020
# (data/raw/estimates/pew.zip). Printed beside the survey's level, never used to set it.
PEW_2020 = {"Muslims": 99.303795, "Religiously unaffiliated": 0.441311, "Christians": 0.246676,
            "Jews": 0.008223}

# note_public's survey figures, measured 2026-09-15 and asserted in main.
NOTE = dict(pooled=10413, named=10375, respondents=10373, nonmuslim=65, christian=12,
            no_religion=28, other=25, ibadi=7, ibadi_medenine=4)


def key(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z]", " ", s)).strip()


def card():
    import pyreadstat

    print("\n  the card by wave (Q1012 labels, the whole file's):")
    for name, _o, _z, sav in ab.WAVES:
        if name not in WAVES and name not in OMIT:
            continue
        meta = pyreadstat.read_sav(os.path.join(ab.AB_DIR, sav), metadataonly=True)[1]
        rel = next(c for c in meta.column_names if c.lower() == "q1012")
        labs = [ab.fold(v) for v in meta.variable_value_labels.get(rel, {}).values()]
        box = any(l in ("atheist", "no religion") for l in labs)
        print(f"    {name:<5} no-religion box: {'yes' if box else 'no'}")
        if box != (name in WAVES):
            raise SystemExit(f"wave {name}: the card does not match the pool; read OMIT")


def contradictions(df):
    sect = df["sect_m"].where(df["sect_m"].notna(), df["sect_c"]).map(
        lambda v: ab.fold(v) if isinstance(v, str) else None)
    cat = df["category"]
    bad = (((cat != "Muslim") & sect.isin(SECT_ISLAM))
           | ((cat == "Muslim") & sect.isin(SECT_CHRISTIAN))
           | (cat.isin(["No religion", "Other"]) & sect.isin(SECT_CHRISTIAN)))
    if bad.any():
        raise SystemExit(f"answers contradicting the follow-up: "
                         f"{df.loc[bad, ['wave', 'category', 'sect_m', 'sect_c']].to_dict('records')}")
    nm = df[cat != "Muslim"]
    fu = {k: sorted(set(x["sect_m"].dropna()) | set(x["sect_c"].dropna()))
          for k, x in nm.groupby(["wave", "category"])}
    print("\n  follow-ups given by non-Muslims (none contradicts): "
          + "; ".join(f"{w} {c}: {v}" for (w, c), v in fu.items() if v))
    if (cat == "Jewish").any():
        raise SystemExit("a Jewish answer is in the pool; read it before adding a category")


def decode(g):
    """INS governorate code per respondent: names in V, VII, VIII; codes in VI (see docstring)."""
    g = g.copy()
    g["number"] = np.nan
    for w, base in NAMED_WAVES.items():
        m = g["wave"] == w
        num = g.loc[m, "k"].map(NORM)
        if num.isna().any():
            raise SystemExit(f"wave {w}: Q1 labels with no governorate: "
                             f"{sorted(set(g.loc[m & num.isna().reindex(g.index, fill_value=False), 'geo_raw']))}")
        pos = (g.loc[m, "geo_code"] - base).astype(int)
        witness = pos.map(lambda p: OFFICIAL_ORDER[p - 1] if 1 <= p <= 24 else None)
        if (witness != num).any():
            raise SystemExit(f"wave {w}: {int((witness != num).sum())} Q1 codes disagree with the name")
        g.loc[m, "number"] = num
    for w in ("VI-1", "VI-2", "VI-3"):
        m = g["wave"] == w
        c = (g.loc[m, "geo_code"] - VI_BASE).astype(int)
        num = c.map(VI_CODES)
        if num.isna().any():
            raise SystemExit(f"wave {w}: Q1 codes outside VI_CODES: {sorted(set(c[num.isna()]))}")
        by_name = g.loc[m, "k"].map(NORM)
        odd = sorted(set(zip(c[by_name != num], g.loc[m, "k"][by_name != num])))
        if odd != sorted(VI_MISLABELLED.items()):
            raise SystemExit(f"wave {w}: codes whose label names another governorate {odd}, "
                             f"not {sorted(VI_MISLABELLED.items())}")
        g.loc[m, "number"] = num
    g["number"] = g["number"].astype(int)
    print("  waves V, VII, VIII decoded by name, every Q1 code in INS order agreeing; VI by code, "
          "every label agreeing except 21009 `Jendouba`")

    # sample-size witness for the Jendouba pair and for VI's whole code table
    vi = g[g["wave"].str.startswith("VI")]
    rest = g[~g["wave"].str.startswith("VI")]
    s_vi = vi["number"].value_counts(normalize=True)
    s_rest = rest["number"].value_counts(normalize=True)
    rho = stats.spearmanr(s_vi.reindex(OFFICIAL_ORDER), s_rest.reindex(OFFICIAL_ORDER)).statistic
    kept = abs(np.log((s_vi[23] / s_vi[22]) / (s_rest[23] / s_rest[22])))
    swap = abs(np.log((s_vi[22] / s_vi[23]) / (s_rest[23] / s_rest[22])))
    print(f"  VI's respondents per governorate against V, VII, VIII's: Spearman {rho:+.3f}; "
          f"Le Kef/Jendouba {s_vi[23] / s_vi[22]:.2f} in VI (21009 as Le Kef) against "
          f"{s_rest[23] / s_rest[22]:.2f} elsewhere, {s_vi[22] / s_vi[23]:.2f} if swapped")
    if rho < 0.95 or kept >= swap:
        raise SystemExit("VI's code table does not reproduce the other waves' sample allocation")
    dup = g.groupby(["wave", "number"])["geo_raw"].nunique()
    if (dup > 1).any():
        raise SystemExit(f"a wave uses two labels for a governorate: {dup[dup > 1].to_dict()}")
    return g


def standouts(g):
    cand = g.groupby("number")["nonm"].sum()
    cand = cand[cand >= 2]
    bar = 0.05 / len(cand)
    print(f"\n  standouts: {len(cand)} governorates with two or more non-Muslims, each against the "
          f"rest in both halves ({' | '.join(','.join(h) for h in HALVES)}), bar P < {bar:.4f}:")
    found = set()
    for n in cand.sort_values(ascending=False).index:
        ps = [stratified_p(d, d["number"] == n) for d in (g[g["wave"].isin(h)] for h in HALVES)]
        print(f"    {n:>2}  " + "   ".join(f"{o} vs {e:.1f} P={p:.3f}" for o, e, p in ps))
        if all(p < bar for _o, _e, p in ps):
            found.add(int(n))
    if found != STANDOUTS:
        raise SystemExit(f"governorates standing apart: {sorted(found)}, not {sorted(STANDOUTS)}")


def greater_tunis(g, a):
    z = g["number"].isin(GREATER_TUNIS)
    o, e, p = stratified_p(g, z)
    hs = [stratified_p(d, d["number"].isin(GREATER_TUNIS)) for d in
          (g[g["wave"].isin(h)] for h in HALVES)]
    za = a["geo_raw"].map(key).isin(AFRO_GREATER_TUNIS)
    oa, ea, pa = stratified_p(a, za)
    print(f"\n  Greater Tunis (Tunis, Ariana, Ben Arous, Manouba) against the rest: Arab Barometer "
          f"{o} vs {e:.1f}, P = {p:.3f}; halves "
          + "; ".join(f"{x} vs {y:.1f} P={q:.3f}" for x, y, q in hs)
          + f"; Afrobarometer {oa} vs {ea:.1f}, P = {pa:.3f}")
    print(f"    weighted: Greater Tunis {wshare(g[z], g.loc[z, 'nonm']):.3%}, the rest "
          f"{wshare(g[~z], g.loc[~z, 'nonm']):.3%}")
    passed = all(q < 0.05 for _x, _y, q in hs) and pa < 0.05
    if passed != GREATER_TUNIS_EXPECTED:
        raise SystemExit(f"Greater Tunis {'now' if passed else 'no longer'} stands apart in both "
                         "halves and both surveys; read sources/tn.md §4 before changing the build")


def afro_witness():
    a = afro.load(COUNTRY, expect_rounds=AFRO_ROUNDS, extra=["URBRUR"])
    a = a.rename(columns={"round": "wave"})
    muslim = a["category"].str.contains(AFRO_MUSLIM)
    other = sorted(set(a.loc[~muslim, "category"]) - AFRO_NON_MUSLIM)
    if other:
        raise SystemExit(f"Afrobarometer answers neither Muslim nor known non-Muslim: {other}")
    a["nonm"] = ~muslim
    if not a["URBRUR"].astype(str).str.lower().isin(["urban", "rural"]).all():
        raise SystemExit("Afrobarometer URBRUR has a value that is neither Urban nor Rural")
    a["urban"] = a["URBRUR"].astype(str).str.lower().eq("urban")
    lvl = pd.Series({r: wshare(x, x["nonm"]) for r, x in a.groupby("wave")})
    print(f"\n  Afrobarometer R5-R9 (witness): {int(a['nonm'].sum())} non-Muslims in {len(a):,}, "
          f"weighted {wshare(a, a['nonm']):.2%}; by round "
          + ", ".join(f"R{r} {v:.2%}" for r, v in lvl.items()))
    print("    " + ", ".join(f"{k} {v}" for k, v in
                             a.loc[a["nonm"], "category"].str.split(" \\(").str[0].value_counts().items()))
    return a


def urban_test(g, a):
    d = g[g["urban"].notna()].copy()
    d["u"] = d["urban"].str.lower().eq("urban")
    o1, e1, p1 = stratified_p(d, d["u"])
    o2, e2, p2 = stratified_p(a, a["urban"])
    print(f"\n  urban excess, exact within wave: Arab Barometer ({','.join(sorted(set(d['wave'])))}) "
          f"{o1} urban of {int(d['nonm'].sum())} against {e1:.1f} expected, P = {p1:.3f}; "
          f"Afrobarometer {o2} of {int(a['nonm'].sum())} against {e2:.1f}, P = {p2:.3f}")
    kn = d[d["nonm"] & d["u"]]
    cells = kn[kn["psu"].notna()].groupby(["wave", "psu"]).size()
    passed = p1 < URBAN_BAR and p2 < URBAN_BAR and cells.max() / len(kn) <= CELL_CAP
    ru = wshare(d[d["u"]], d.loc[d["u"], "nonm"])
    rr = wshare(d[~d["u"]], d.loc[~d["u"], "nonm"])
    print(f"    weighted: urban {ru:.3%}, rural {rr:.3%}")
    if passed != URBAN_EXPECTED:
        raise SystemExit(f"the urban test {'passes' if passed else 'fails'}, and the build was "
                         "written for the other outcome; read sources/tn.md §4 before changing it")


def vii_cluster(g):
    vo = g[(g["wave"] == "VII") & (g["category"] == "Other")]
    inside = int(vo["number"].isin(VII_CLUSTER).sum())
    print(f"\n  wave VII `Other`: {len(vo)} answers, {inside} in Le Kef, Siliana and Sousse "
          f"(PSUs {sorted(set(vo.loc[vo['number'].isin(VII_CLUSTER), 'psu'].astype(int)))})")
    if (inside, len(vo)) != VII_CLUSTER_OTHER:
        raise SystemExit(f"wave VII's Other cluster is {(inside, len(vo))}, not {VII_CLUSTER_OTHER}")
    early = g[g["wave"].isin(HALVES[0]) & g["number"].isin(VII_CLUSTER)]
    print(f"    those three governorates in waves V to VI-3: {int(early['nonm'].sum())} non-Muslims "
          f"in {len(early)}")
    drop = vo.index[vo["number"].isin(VII_CLUSTER)]
    h = g.drop(index=drop)
    hn = h[h["nonm"]]
    mix = (hn.groupby("category")["w"].sum() / hn["w"].sum()).reindex(NON_MUSLIM, fill_value=0)
    print(f"    kept in the pool; without them the share would be {wshare(h, h['nonm']):.3%} and the "
          "mix " + ", ".join(f"{c} {v:.1%}" for c, v in mix.items()))


def main():
    card()
    print("\n=== Arab Barometer, Tunisia ===")
    df = ab.load(COUNTRY, expect_waves=WAVES, waves=WAVES, omit=OMIT, recode=RECODE,
                 extra={"sect_m": ("Q1012A_MUSLIM", "q1012a"), "sect_c": ("Q1012A_CHRISTIAN",),
                        "urban": ("q13",)},
                 raw={"psu": ("psu",)})
    pooled = len(df)
    print(pd.crosstab(df["category"], df["wave"]).to_string())
    for cat, why in DROPPED.items():
        print(f"  dropping {int((df['category'] == cat).sum())} {cat!r}: {why}")
        df = df[df["category"] != cat]
    named = len(df)
    contradictions(df)

    g = df.copy()
    g["k"] = g["geo_raw"].map(key)
    bogus = g["k"].isin(BOGUS)
    if (g.loc[bogus, "category"] != "Muslim").any():
        raise SystemExit("a respondent with no governorate is not Muslim")
    print(f"  {int(bogus.sum())} respondents refused a governorate (all Muslim) and leave the "
          "geography")
    g = decode(g[~bogus])

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if sorted(lut["code"]) != sorted(OFFICIAL_ORDER):
        raise SystemExit("tn_lookup.csv is not the 24 governorates")
    if g["number"].nunique() != 24:
        raise SystemExit(f"the pool samples {g['number'].nunique()} governorates, not 24")
    g["geo_id"] = g["number"].map(lambda n: f"TN{n}")
    pop = lut.set_index("geo_id")["pop"]
    ab.held_out(g, pop, COUNTRY, pop_source="RGPH 2024")
    ab.assert_not_quota(g, COUNTRY)

    g["nonm"] = g["category"] != "Muslim"
    standouts(g)
    a = afro_witness()
    greater_tunis(g, a)
    urban_test(g, a)
    vii_cluster(g)

    L = wshare(g, g["nonm"])
    print(f"\n  weighted non-Muslim share, waves V-VIII: {L:.3%}; by wave "
          + ", ".join(f"{w} {wshare(x, x['nonm']):.2%}" for w, x in g.groupby('wave', sort=False)))
    if not NONMUSLIM_BAND[0] <= L <= NONMUSLIM_BAND[1]:
        raise SystemExit(f"{L:.3%} is outside {NONMUSLIM_BAND}")
    nm = g[g["nonm"]]
    comp = (nm.groupby("category")["w"].sum() / nm["w"].sum()).reindex(NON_MUSLIM, fill_value=0.0)
    print("    non-Muslim mix (weighted): " + ", ".join(
        f"{c} {v:.1%} (n={int((nm['category'] == c).sum())})" for c, v in comp.items()))
    print("  outside level check, Pew Research Center 2020: "
          + ", ".join(f"{k} {v:.3f}%" for k, v in PEW_2020.items())
          + f"; the survey reads Christian {100 * comp['Christian'] * L:.3f}% and no religion "
          f"{100 * comp['No religion'] * L:.3f}%")

    ibadi = g[g["sect_m"].map(lambda v: isinstance(v, str) and ab.fold(v) in ("ibadi", "ibadhi"))]
    print(f"\n  witness, not drawn: Ibadi on the follow-up, {len(ibadi)} in the pool: "
          f"{ibadi.groupby(['wave', 'geo_raw']).size().to_dict()}")

    got = dict(pooled=pooled, named=named, respondents=len(g), nonmuslim=len(nm),
               christian=int((nm["category"] == "Christian").sum()),
               no_religion=int((nm["category"] == "No religion").sum()),
               other=int((nm["category"] == "Other").sum()),
               ibadi=len(ibadi), ibadi_medenine=int((ibadi["number"] == 52).sum()))
    print(f"\n  note_public's survey figures: {got}")
    if NOTE and got != NOTE:
        raise SystemExit(f"note_public's figures are {NOTE}; the pool gives {got}")

    names = dict(zip(lut["geo_id"], lut["name"]))
    m = pd.DataFrame({"Muslim": pop * (1 - L), **{c: pop * L * comp[c] for c in NON_MUSLIM}})[CATS]
    counts = round_within_rows(m)
    if not (counts.sum(axis=1) == pop.reindex(counts.index)).all():
        raise SystemExit("a governorate's rounded counts do not sum to its census population")
    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_id", "source_category", "count"]
    n_by = g.groupby("geo_id").size()
    out["geo_level"] = "governorate"
    out["geo_name"] = out["geo_id"].map(names)
    out["basis"] = "self_id"
    out["year"] = YEARS
    out["source_id"] = SOURCE_ID
    out["note"] = out["geo_id"].map(
        lambda i: f"Arab Barometer waves V to VIII pooled, n={int(n_by[i])} in this governorate; "
                  "the national non-Muslim share and mix, on the RGPH 2024 count")
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
            "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    total = int(out["count"].sum())
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, 24 governorates)")
    drawn = out.groupby("source_category")["count"].sum()
    for c in CATS:
        print(f"    {drawn[c] / total:8.3%}  {c}  ({int(drawn[c]):,})")


if __name__ == "__main__":
    main()
