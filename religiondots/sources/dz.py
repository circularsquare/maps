"""Algeria: religion by wilaya from the pooled Arab Barometer, with Kabylie as one unit.

Reads data/raw/arabbarometer/*.sav and data/geo/dz/dz_lookup.csv; writes data/normalized/dz.csv.
`sources/dz.md` is the record in prose; `sources.md` §maghreb-2026-09-16 is the scouting it
builds on; `ask/RULINGS.md` (2026-09-15 and 2026-09-16) has Anita's two rulings.

## WHAT IS DRAWN

No Algerian census has asked religion (the 2008 and 2022 forms were read, `sources.md`
§maghreb). Arab Barometer waves V to VII ask it of about 7,700 adults. Two things are measured and
drawn:

  * **the non-Muslim share, in two places**: the three Kabyle wilayas (Tizi Ouzou, Béjaïa,
    Bouira) as one unit, on Anita's ruling of 2026-09-16, and the other 45 wilayas as a second
    unit. `kabylie_test` re-measures the contrast on every build and stops if it no longer holds;
    `standouts` asks whether any single wilaya outside Kabylie stands apart, and none does.
  * **what the non-Muslims said**, Christian, no religion, or other, at the pooled national
    composition inside each unit's non-Muslim share (spec §12 "SMALL CATEGORIES GO IN THE
    RESIDUAL"; 38 respondents cannot carry a composition per unit).

Muslim is each wilaya's remainder. The shares go onto the 2008 census count of each wilaya, the
last one ONS has published (`sources/dz_geo.py`).

## THE POOL IS WAVES V TO VII, BECAUSE THE CARD CHANGES

`card()` reads each wave's `Q1012` labels. Waves II, III and IV offer no box for having no
religion; V offers `Atheist`, VI-1, VI-2, VI-3 and VII offer `No religion`. Nearly half of
Algeria's non-Muslim answers are that box, so a share pooled with II-IV measures which
questionnaire was used (`playbooks/arabbarometer.md`, "The card changes by wave"). II-IV are
left out with that reason in `OMIT`; between them they hold one non-Muslim in 3,636 answers.
`Atheist` (V) and `No religion` (VI-VII) are merged in `RECODE` as each card's one box for having
none: both are code 4 on their card, and no card offers both.

## TWO ANSWERS THAT CONTRADICT EACH OTHER LEAVE THE POOL

Wave V asks a denomination item (`Q1012A`) of everyone who names a religion, on one list holding
Muslim schools and Christian churches. Six Algerian answers contradict `Q1012`: all three
`Jewish` answers give Shafi'i, Sunni or Orthodox, and three `Christian` answers give Shia, Sunni
or `Just a Muslim`. Which of the two items was mis-keyed cannot be told, so the six leave the pool
as a refusal does (`contradictions`, asserted). With them out, no Algerian in the pool answered
Jewish.

## THE WAVE VI WEIGHTS

All three parts of wave VI carry Algeria's `WT` un-normalised (means 0.840, 0.780, 0.849; every
other country in those files averages 0.99-1.00), so `ab.load` is told to divide them by their
mean (`RESCALE`). Shares within the wave do not move; the wave then counts for its interviews in
the pool.

Usage:
    python sources/dz.py            rebuild data/normalized/dz.csv
"""

import os
import re
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

import arabbarometer as ab
from afrobarometer import round_within_rows
from stability import CELL_CAP

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOOKUP = os.path.join(ROOT, "data", "geo", "dz", "dz_lookup.csv")
OUT = os.path.join(ROOT, "data", "normalized", "dz.csv")

COUNTRY = "Algeria"
WAVES = ["V", "VI-1", "VI-2", "VI-3", "VII"]
YEARS = "2018-2022"
SOURCE_ID = "dz_arabbarometer_2018_2022"

_NO_BOX = ("the card has no box for having no religion, and nearly half of Algeria's non-Muslim "
           "answers in the waves that offer one are that box; pooling this wave would mix two "
           "questionnaires (sources/dz.md §3)")
OMIT = {"II": _NO_BOX, "III": _NO_BOX, "IV": _NO_BOX}

_WT = ("Algeria's WT in wave VI is labelled a probability-of-selection weight and was left "
       "un-normalised: every other country in the same file averages 0.99-1.00 (sources/dz.md §3)")
RESCALE = {"VI-1": _WT, "VI-2": _WT, "VI-3": _WT}

RECODE = {
    "refused": "Refused to answer",      # wave V
    "Refused": "Refused to answer",      # wave VI-1
    "Atheist": "No religion",            # wave V: code 4, the card's one box for having none
}
DROPPED = {"Refused to answer": "a refusal is not a religion"}

CATS = ["Muslim", "Christian", "No religion", "Other"]
NON_MUSLIM = ["Christian", "No religion", "Other"]

# The denomination follow-up's answers, by the religion they belong to. A non-Muslim answer whose
# follow-up is a label in neither set stops the build, so a new label is read before it is used.
SECT_ISLAM = {"sunni", "shia", "just a muslim", "shafi'i", "hanbali", "maliki", "hanafi", "ibadi",
              "ibadhi", "mozabite", "ja'fari", "ahmadiyya"}
SECT_CHRISTIAN = {"catholic", "orthodox", "coptic", "protestant", "just a christian",
                  "evangelical"}
# (wave, Q1012 answer, follow-up answer) for each contradiction, as measured 2026-09-15.
CONTRADICTIONS = sorted([
    ("V", "Christian", "just a muslim"), ("V", "Christian", "shia"), ("V", "Christian", "sunni"),
    ("V", "Jewish", "orthodox"), ("V", "Jewish", "shafi'i"), ("V", "Jewish", "sunni"),
])

KABYLIE = {6, 10, 15}                 # Béjaïa, Bouira, Tizi Ouzou
HALVES = [["V"], ["VI-1", "VI-2", "VI-3", "VII"]]
STANDOUTS = set()                     # wilayas outside Kabylie drawn apart; asserted
# The survey's weighted national non-Muslim share must sit in this band. Set after the probe
# (unweighted, waves V-VII: 36 of about 7,700 after the contradictions), so it guards a re-release
# rather than testing this build blind.
NONMUSLIM_BAND = (0.001, 0.012)

# official wilaya number -> every Q1 spelling in waves II to VII, folded by `key`.
SPELLINGS = {
    1: ["adrar"], 2: ["chlef"], 3: ["laghouat"],
    4: ["oum el bouaghi", "o e bouaghi", "oeb"], 5: ["batna"], 6: ["bejaia", "bejia"],
    7: ["biskra", "beskra", "biskara"], 8: ["bechar", "bashar"], 9: ["blida"], 10: ["bouira"],
    11: ["tamanrasset", "tamanghasset", "tamenrasset"], 12: ["tebessa", "tbessa"],
    13: ["tlemcen"], 14: ["tiaret"], 15: ["tizi ouzou", "tiz ouzou"], 16: ["algiers"],
    17: ["djelfa"], 18: ["jijel"], 19: ["setif"], 20: ["saida"], 21: ["skikda"],
    22: ["sidi bel abbes", "sidi b abbes"], 23: ["annaba"], 24: ["guelma"],
    25: ["constantine"], 26: ["medea"], 27: ["mostaganem"],
    28: ["m sila", "masila", "messilia"], 29: ["mascara", "musker"], 30: ["ouargla"],
    31: ["oran"], 32: ["el bayadh"], 33: ["illizi"],
    34: ["bordj bou arreridj", "b b arreridj", "bba"], 35: ["boumerdes", "boumderdes"],
    36: ["el taref", "taref", "al traf", "el tarf"], 37: ["tindouf"], 38: ["tissemsilt"],
    39: ["el oued", "al wad"], 40: ["khenchela"], 41: ["souk ahras"], 42: ["tipaza", "tipasa"],
    43: ["mila"], 44: ["ain defla"], 45: ["naama"],
    46: ["ain temouchent", "ain tecmouchent", "timouchent"], 47: ["ghardaia"], 48: ["relizane"],
}
NORM = {s: n for n, ss in SPELLINGS.items() for s in ss}
BOGUS = {"don t know"}                # VI-2 and VI-3 code 99998; every one answered Muslim

# Where the Q1 CODE decodes to the official wilaya number. Wave VII's codes are its own order.
CODE_SYSTEMS = {"V": 10000, "VI-1": 1000, "VI-2": 1000, "VI-3": 1000}

# note_public's survey figures, which the CSV cannot carry; asserted in `note_figures`.
NOTE = dict(pooled=7699, named=7676, respondents=7653, kabylie_n=613, kabylie_nonm=16,
            rest_n=7040, rest_nonm=20, christian=14, no_religion=19, other=3)

# Pew Research Center, Religious Composition by Country 2010-2020, percentages file, Algeria 2020
# (data/raw/estimates/pew.zip). Printed beside the survey's level, never used to set it.
PEW_2020 = {"Muslims": 0.9838234711, "Religiously unaffiliated": 0.01266207099,
            "Christians": 0.00294989556, "Other religions": 0.00041328549,
            "Hindus": 0.00015000784, "Jews": 0.0000012999}


def key(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"^\s*\d+\s*[.)]\s*", "", s).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z]", " ", s)).strip()


def card():
    """Each wave's Q1012 labels, and whether the card offers a box for having no religion."""
    import pyreadstat

    print("\n  the card by wave (Q1012 value labels as each file declares them):")
    for name, _o, _z, sav in ab.WAVES:
        if name == "I" or name == "VIII":
            continue
        meta = pyreadstat.read_sav(os.path.join(ab.AB_DIR, sav), metadataonly=True)[1]
        rel = next(c for c in meta.column_names if c.lower() == "q1012")
        labs = [ab.fold(v) for v in meta.variable_value_labels.get(rel, {}).values()]
        none_box = any(l in ("atheist", "no religion") for l in labs)
        print(f"    {name:<5} none box: {'yes' if none_box else 'no ':<4} {labs}")
        if none_box != (name in WAVES):
            raise SystemExit(f"wave {name}: a no-religion box is {'on' if none_box else 'off'} "
                             "the card, which is not what the pool assumes; read OMIT")


def contradictions(df):
    """Drop respondents whose religion answer contradicts their denomination answer."""
    sect = df["sect_m"].where(df["sect_m"].notna(), df["sect_c"]).map(
        lambda v: ab.fold(v) if isinstance(v, str) else None)
    cat = df["category"]
    known = SECT_ISLAM | SECT_CHRISTIAN
    odd = df[(cat != "Muslim") & sect.notna() & ~sect.isin(known)
             & ~sect.isin({"other", "don't know", "refused", "refused to answer",
                           "decline to answer"})]
    if len(odd):
        raise SystemExit(f"non-Muslim answers with an unclassified denomination: "
                         f"{odd[['wave', 'category', 'sect_m', 'sect_c']].to_dict('records')}")
    bad = (((cat == "Muslim") & sect.isin(SECT_CHRISTIAN))
           | ((cat == "Christian") & sect.isin(SECT_ISLAM))
           | (cat.isin(["Jewish", "No religion", "Other"]) & sect.isin(known)))
    found = sorted(zip(df.loc[bad, "wave"], cat[bad], sect[bad]))
    print(f"\n  answers contradicting the denomination follow-up: {len(found)}")
    for w, c, s in found:
        print(f"    wave {w}: {c} / {s}")
    if found != CONTRADICTIONS:
        raise SystemExit(f"the contradictions are {found}, not {CONTRADICTIONS}")
    out = df[~bad].copy()
    if (out["category"] == "Jewish").any():
        raise SystemExit("a Jewish answer survives the contradiction check; read it before "
                         "adding a Jewish category")
    return out


def stratified_p(d, in_zone):
    """(observed, expected, P(X >= observed)) for non-Muslims inside `in_zone`, exact within wave."""
    dist, obs, exp = np.array([1.0]), 0, 0.0
    for _w, x in d.groupby("wave"):
        N, K, n = len(x), int(x["nonm"].sum()), int(in_zone.loc[x.index].sum())
        obs += int((x["nonm"] & in_zone.loc[x.index]).sum())
        exp += n * K / N if N else 0.0
        lo, hi = max(0, n - (N - K)), min(n, K)
        pmf = np.zeros(hi + 1)
        ks = np.arange(lo, hi + 1)
        pmf[lo:] = hypergeom.pmf(ks, N, K, n)
        dist = np.convolve(dist, pmf)
    return obs, exp, float(dist[obs:].sum()) if obs < len(dist) else 0.0


def wshare(d, mask):
    return float(d.loc[mask, "w"].sum() / d["w"].sum())


def kabylie_test(g):
    """The ruling's contrast, re-measured: pooled, in each half of the waves, and per PSU."""
    kab = g["number"].isin(KABYLIE)
    print("\n  Kabylie (Tizi Ouzou, Béjaïa, Bouira) against the other 45 wilayas, by wave:")
    print(f"    {'wave':<6}{'Kabylie':>18}{'rest':>18}")
    for w in WAVES:
        x = g[g["wave"] == w]
        k, r = x[kab.loc[x.index]], x[~kab.loc[x.index]]
        print(f"    {w:<6}{int(k['nonm'].sum()):>6} of {len(k):>5}    {int(r['nonm'].sum()):>6} of "
              f"{len(r):>5}")
    obs, exp, p = stratified_p(g, kab)
    print(f"    pooled: {obs} non-Muslims in Kabylie against {exp:.1f} expected, exact within "
          f"wave, P = {p:.2g}")
    for half in HALVES:
        d = g[g["wave"].isin(half)]
        o, e, ph = stratified_p(d, kab.loc[d.index])
        print(f"    waves {','.join(half):<18} {o} against {e:.1f} expected, P = {ph:.2g}")
        if ph >= 0.05:
            raise SystemExit(f"Kabylie's excess does not replicate in waves {half}; the ruling's "
                             "unit no longer stands on these data. Stop and read.")
    kn = g[kab & g["nonm"]]
    cells = kn[kn["psu"].notna()].groupby(["wave", "psu"]).size()
    top = cells.max() / len(kn)
    print(f"    the largest PSU holds {int(cells.max())} of Kabylie's {len(kn)} non-Muslims "
          f"({top:.0%}; cap {CELL_CAP:.0%}); they come from {kn['number'].nunique()} wilayas and "
          f"{len(cells)} PSUs in the waves that carry one")
    if top > CELL_CAP or kn["number"].nunique() < 2:
        raise SystemExit("Kabylie's non-Muslims sit in one sampling cell or one wilaya")
    s_k, s_r = wshare(g[kab], g.loc[kab, "nonm"]), wshare(g[~kab], g.loc[~kab, "nonm"])
    print(f"    weighted: Kabylie {s_k:.2%}, the rest {s_r:.2%}")
    return s_k, s_r


def standouts(g):
    """Does any wilaya outside Kabylie stand apart? Replicated in both halves, Bonferroni."""
    rest = g[~g["number"].isin(KABYLIE)]
    cand = rest.groupby("number")["nonm"].sum()
    cand = cand[cand >= 2]
    bar = 0.05 / len(cand)
    print(f"\n  standouts outside Kabylie: {len(cand)} wilayas with two or more non-Muslims, each "
          f"against the other 44\n  in both halves of the waves, bar P < {bar:.4f} (Bonferroni) in "
          "each:")
    found = set()
    for n in cand.index:
        ps = []
        for half in HALVES:
            d = rest[rest["wave"].isin(half)]
            ps.append(stratified_p(d, d["number"] == n))
        stands = all(p < bar for _o, _e, p in ps)
        print(f"    {n:>2}  " + "   ".join(f"{o} vs {e:.1f} P={p:.3f}" for o, e, p in ps)
              + ("   STANDS APART" if stands else ""))
        if stands:
            found.add(int(n))
    if found != STANDOUTS:
        raise SystemExit(f"the wilayas standing apart are {sorted(found)}, not {sorted(STANDOUTS)}")


def code_witness(g):
    for w, base in CODE_SYSTEMS.items():
        sub = g[g["wave"] == w]
        bad = sub[(sub["geo_code"] - base) != sub["number"]]
        print(f"    wave {w:<5} code - {base} = official number: {len(sub) - len(bad)} agree, "
              f"{len(bad)} do not")
        if len(bad):
            raise SystemExit(f"wave {w}: the Q1 code and the harmonised name disagree on "
                             f"{len(bad)} respondents")


def note_figures(pooled, named, g):
    kab = g["number"].isin(KABYLIE)
    got = dict(pooled=pooled, named=named, respondents=len(g),
               kabylie_n=int(kab.sum()), kabylie_nonm=int(g.loc[kab, "nonm"].sum()),
               rest_n=int((~kab).sum()), rest_nonm=int(g.loc[~kab, "nonm"].sum()),
               christian=int((g["category"] == "Christian").sum()),
               no_religion=int((g["category"] == "No religion").sum()),
               other=int((g["category"] == "Other").sum()))
    print(f"\n  note_public's survey figures: {got}")
    if NOTE and got != NOTE:
        raise SystemExit(f"note_public's survey figures are {NOTE}; the pool now gives {got}. "
                         "Fix the note and NOTE together.")


def main():
    card()
    print("\n=== Arab Barometer, Algeria ===")
    df = ab.load(COUNTRY, expect_waves=WAVES, waves=WAVES, omit=OMIT, recode=RECODE,
                 extra={"sect_m": ("q1012a", "q1012a_muslim"), "sect_c": ("q1012a_christian",)},
                 raw={"psu": ("psu",)}, rescale_weights=RESCALE)
    pooled = len(df)
    print(f"\n  pooled: {pooled:,} respondents with a religion answer")
    print(pd.crosstab(df["category"], df["wave"]).to_string())

    for cat, why in DROPPED.items():
        n = int((df["category"] == cat).sum())
        print(f"  dropping {n} who answered {cat!r}: {why}")
        df = df[df["category"] != cat]
    named = len(df)
    df = contradictions(df)

    g = df.copy()
    g["k"] = g["geo_raw"].map(key)
    bogus = g["k"].isin(BOGUS)
    if bogus.any():
        if (g.loc[bogus, "category"] != "Muslim").any():
            raise SystemExit("a respondent with no wilaya is not Muslim; they cannot be dropped "
                             "without moving a count")
        print(f"\n  {int(bogus.sum())} respondents have `Don't know` for a wilaya (waves "
              f"{sorted(set(g.loc[bogus, 'wave']))}), all Muslim; dropped from the geography")
        g = g[~bogus]
    g["number"] = g["k"].map(NORM)
    unmapped = sorted(set(g.loc[g["number"].isna(), "geo_raw"]))
    if unmapped:
        raise SystemExit(f"Q1 labels with no wilaya: {unmapped}")
    g["number"] = g["number"].astype(int)
    dup = g.groupby(["wave", "number"])["geo_raw"].nunique()
    if (dup > 1).any():
        raise SystemExit(f"one wave uses two labels for a wilaya: {dup[dup > 1].to_dict()}")
    print(f"  {g['geo_raw'].nunique()} Q1 labels -> {g['number'].nunique()} wilayas")
    code_witness(g)

    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str})
    if sorted(lut["number"]) != list(range(1, 49)):
        raise SystemExit("dz_lookup.csv is not the 48 wilayas")
    if g["number"].nunique() != 48:
        raise SystemExit(f"the pool samples {g['number'].nunique()} wilayas, not 48; a wilaya "
                         "no wave reached would be drawn at its unit's rate without a word")
    gid = dict(zip(lut["number"], lut["geo_id"]))
    names = dict(zip(lut["geo_id"], lut["name"]))
    g["geo_id"] = g["number"].map(gid)
    pop = lut.set_index("geo_id")["pop"]
    ab.held_out(g, pop, COUNTRY, pop_source="RGPH 2008")
    ab.assert_not_quota(g, COUNTRY)

    g["nonm"] = g["category"] != "Muslim"
    s_kab, s_rest = kabylie_test(g)
    standouts(g)

    nat = wshare(g, g["nonm"])
    print(f"\n  weighted national non-Muslim share in the survey: {nat:.3%}")
    if not NONMUSLIM_BAND[0] <= nat <= NONMUSLIM_BAND[1]:
        raise SystemExit(f"{nat:.3%} is outside {NONMUSLIM_BAND}")
    nm = g[g["nonm"]]
    comp = nm.groupby("category")["w"].sum() / nm["w"].sum()
    comp = comp.reindex(NON_MUSLIM, fill_value=0.0)
    kab = g["number"].isin(KABYLIE)
    for lab, part in (("Kabylie", nm[nm["number"].isin(KABYLIE)]),
                      ("rest", nm[~nm["number"].isin(KABYLIE)])):
        c = part.groupby("category")["w"].sum() / part["w"].sum()
        print(f"    composition in {lab}: " + ", ".join(f"{k} {v:.0%} (n={int((part['category'] == k).sum())})"
                                               for k, v in c.items()))
    print("    drawn composition (national, weighted): "
          + ", ".join(f"{k} {v:.1%}" for k, v in comp.items()))

    ibadi = df[df["sect_m"].map(lambda v: isinstance(v, str) and ab.fold(v) in
                                ("ibadi", "ibadhi", "mozabite"))]
    print(f"\n  witness, not drawn: Ibadi or Mozabite on the follow-up, {len(ibadi)} in the pool: "
          f"{ibadi.groupby(['wave', 'geo_raw']).size().to_dict()}")
    print("\n  outside level check, Pew Research Center 2020 (Religious Composition 2010-2020): "
          + ", ".join(f"{k} {v:.2%}" for k, v in PEW_2020.items()))
    print(f"    the survey reads Christian {comp['Christian'] * nat:.2%} and no religion "
          f"{comp['No religion'] * nat:.2%}: a floor for a question put by an interviewer")
    note_figures(pooled, named, g)

    rows = {}
    for r in lut.itertuples():
        s = s_kab if r.number in KABYLIE else s_rest
        nonm = r.pop * s
        rows[r.geo_id] = {"Muslim": r.pop - nonm, **{c: nonm * comp[c] for c in NON_MUSLIM}}
    m = pd.DataFrame.from_dict(rows, orient="index")[CATS]
    counts = round_within_rows(m)
    if not (counts.sum(axis=1) == pop.reindex(counts.index)).all():
        raise SystemExit("a wilaya's rounded counts do not sum to its census population")

    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_id", "source_category", "count"]
    n_by = g.groupby("geo_id").size()
    out["geo_level"] = "wilaya"
    out["geo_name"] = out["geo_id"].map(names)
    out["basis"] = "self_id"
    out["year"] = YEARS
    out["source_id"] = SOURCE_ID
    out["note"] = out["geo_id"].map(
        lambda i: (f"Arab Barometer waves V to VII pooled, n={int(n_by[i])} in this wilaya; "
                   + ("Kabylie's" if int(i[2:]) in KABYLIE else "the other 45 wilayas'")
                   + " non-Muslim share, national composition, on the RGPH 2008 count"))
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
            "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")
    total = int(out["count"].sum())
    print(f"\nwrote {OUT} ({len(out)} rows, {total:,} people, 48 wilayas)")
    drawn = out.groupby("source_category")["count"].sum()
    for c in CATS:
        print(f"    {drawn[c] / total:8.3%}  {c}  ({int(drawn[c]):,})")
    kab_ids = [gid[n] for n in sorted(KABYLIE)]
    kc = out[out["geo_id"].isin(kab_ids)]
    print(f"    Kabylie: {int(kc[kc['source_category'] != 'Muslim']['count'].sum()):,} non-Muslims "
          f"of {int(kc['count'].sum()):,}; the rest: "
          f"{int(out[~out['geo_id'].isin(kab_ids) & (out['source_category'] != 'Muslim')]['count'].sum()):,}")


if __name__ == "__main__":
    main()
