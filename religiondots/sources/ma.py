"""Morocco: religion from the pooled Arab Barometer on the 2024 census, and foreign residents by
nationality.

Reads data/raw/arabbarometer/*.sav, data/raw/afrobarometer/ (a witness), data/geo/ma/ma_pieces.csv
and data/raw/estimates/pew.zip; writes data/normalized/ma.csv (Moroccans) and
data/normalized/ma_foreign.csv (foreign residents). `sources/ma.md` is the record in prose;
`sources.md` §maghreb-2026-09-16 the scouting; `ask/RULINGS.md` 2026-09-15 and 2026-09-16 the rulings.

## TWO POPULATIONS, ONE CENSUS TABLE

HCP's 2024 legal-population workbook prints Moroccans and foreigners separately for every unit,
so the two halves partition the country by construction, as Greece's do (`sources/gr.py`):

  * **Moroccans, 36,680,178.** The Arab Barometer interviews citizens aged 18 and over (the
    technical reports for waves V, VII and VIII say so). No Moroccan census asks religion (the
    2014 and 2024 forms were read, `sources.md` §11aq).
  * **Foreign residents, 148,152.** Counted per commune; their nationality is published only
    nationally (HCP, *Les résidents étrangers au Maroc*, November 2025). Each unit's foreigners
    take that national mix, and each nationality Pew's 2020 composition
    (`taxonomy/origin_religion.py`).

## WHAT IS DRAWN FOR MOROCCANS

  * **the non-Muslim share, urban and rural.** No region stands apart (`standouts`). Non-Muslims
    are more urban than the sample in the Arab Barometer and again in the Afrobarometer, an
    independent survey (`urban_test`, pre-registered bar in both), so each unit's Moroccans take
    an urban and a rural share, from the census's own urban and rural counts.
  * **what the non-Muslims said**, Christian, no religion or other, at the pooled national mix.

## THE POOL IS WAVES V TO VIII

`card()` reads each wave's `Q1012` labels. Waves III and IV have no box for having no religion,
which is 15 of the 36 non-Muslim answers where offered, and they sample the 16 regions abolished in
2015. V offers `Atheist`, VI-1 to VIII `No religion`.

Usage:
    python sources/ma.py            rebuild data/normalized/ma.csv and ma_foreign.csv
"""

import io
import os
import re
import sys
import unicodedata
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

import arabbarometer as ab
import afrobarometer as afro
from afrobarometer import round_within_rows
from stability import CELL_CAP

PIECES = os.path.join(ROOT, "data", "geo", "ma", "ma_pieces.csv")
LOOKUP = os.path.join(ROOT, "data", "geo", "ma", "ma_lookup.csv")
PEW = os.path.join(ROOT, "data", "raw", "estimates", "pew.zip")
OUT = os.path.join(ROOT, "data", "normalized", "ma.csv")
OUT_FOREIGN = os.path.join(ROOT, "data", "normalized", "ma_foreign.csv")

COUNTRY = "Morocco"
WAVES = ["V", "VI-1", "VI-2", "VI-3", "VII", "VIII"]
YEARS = "2018-2024"
SOURCE_ID = "ma_arabbarometer_2018_2024"

_OLD = ("the card has no box for having no religion (15 of the 36 non-Muslim answers in V-VIII "
        "are that box) and the wave samples the 16 regions abolished in 2015 (sources/ma.md §3)")
OMIT = {"III": _OLD, "IV": _OLD}
RECODE = {
    "refused": "Refused to answer",                         # wave V
    "Refused": "Refused to answer",                         # wave VI-1
    "Something else: SPECIFY_______": "Other",              # wave VI-3's card has no plain Other
}
DROPPED = {"Refused to answer": "a refusal is not a religion"}
CATS = ["Muslim", "Christian", "No religion", "Other"]
NON_MUSLIM = ["Christian", "No religion", "Other"]
SECT_ISLAM = {"sunni", "shia", "just a muslim", "shafi'i", "hanbali", "maliki", "hanafi",
              "ibadi", "sufi", "ja'fari", "ahmadiyya"}

# HCP region code -> every Q1 spelling in waves V to VIII, folded by `key`.
SPELLINGS = {1: ["tanger tetouan al hoceima"], 2: ["oriental"], 3: ["fes meknes"],
             4: ["rabat sale kenitra"], 5: ["beni mellal khenifra"],
             6: ["grand casablanca settat"], 7: ["marrakech safi"], 8: ["draa tafilalet"],
             9: ["souss massa", "sousse massa"], 10: ["guelmim oued noun"],
             11: ["laayoune sakia el hamra"], 12: ["eddakhla oued eddahab"]}
NORM = {s: n for n, ss in SPELLINGS.items() for s in ss}
BOGUS = {"don t know", "refused"}                  # VI-2 and VI-3; must all be Muslim
CODE_SYSTEMS = {"V": 130000, "VI-1": 13000, "VI-2": 13000, "VI-3": 13000, "VII": 130000,
                "VIII": 130000}                    # Q1 code - base = HCP's region code

HALVES = [["V", "VI-1", "VI-2", "VI-3"], ["VII", "VIII"]]
STANDOUTS = set()                                  # regions drawn apart; asserted
# Pre-registered before the test was run (2026-09-15): the urban excess must reach P < 0.05,
# exact within wave, in the Arab Barometer waves that record the stratum AND in the Afrobarometer.
URBAN_BAR = 0.05
URBAN_EXPECTED = True
NONMUSLIM_BAND = (0.001, 0.012)
AFRO_ROUNDS = [5, 6, 7, 8, 9]
AFRO_MUSLIM = re.compile(r"(?i)muslim|sunni|shia|isma|qadiri|tijani")
AFRO_NON_MUSLIM = {                                # every non-Muslim label, so a new one stops
    "Agnostic", "Agnostic (Do not know if there is a God)",
    "Agnostic(Do not know if there is a God)", "Atheist (Do not believe in a God)",
    "Atheist(Do not believe in a God)", "Calvinist", "Christian only",
    "Christian only (i.e., respondents says only “Christian”, without identifying a specific "
    "sub-group)", "Jewish", "None"}

# note_public's survey figures, measured 2026-09-15 and asserted in main.
NOTE = dict(pooled=10423, named=10398, respondents=10382, nonmuslim=36, christian=19,
            no_religion=15, other=2)

# Foreign residents by nationality, % of all 148,152, HCP (November 2025): regions of nationality
# p.4 and annex tables 1-6 (pp.20-21) as % within the region; Cameroon's 1.9% of all is in the
# text (p.5) and comes out of "other sub-Saharan". "_other" takes Pew's regional total.
MIX = {
    "Sub-Saharan Africa": (59.9, {"SN": 30.8, "CI": 28.9, "GN": 8.0, "ML": 4.1, "CG": 3.9,
                                  "_other": 24.2}),
    "Europe": (20.3, {"FR": 68.2, "ES": 7.0, "IT": 4.0, "BE": 3.2, "DE": 2.7, "_other": 14.9}),
    "Middle East-North Africa": (7.3, {"SY": 41.4, "EG": 13.6, "SA": 9.9, "PS": 7.6, "IQ": 6.7,
                                       "_other": 20.8}),
    "Maghreb": (6.0, {"MR": 31.7, "DZ": 31.4, "TN": 28.9, "LY": 7.9}),
    "Asia-Pacific": (4.1, {"CN": 24.8, "PH": 21.5, "TR": 15.0, "IN": 11.7, "KR": 5.9,
                           "_other": 21.1}),
    "North America": (1.8, {"US": 75.6, "CA": 24.4}),
}
CAMEROON_OF_ALL = 1.9
TEXT_CHECK = {"SN": 18.4, "CI": 17.3, "FR": 13.8, "GN": 4.8, "ML": 2.5, "CG": 2.3, "SY": 3.0}


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


def stratified_p(d, zone, flag="nonm", by="wave"):
    """(observed, expected, P(X >= observed)) of `flag` inside `zone`, exact within `by`."""
    dist, obs, exp = np.array([1.0]), 0, 0.0
    for _w, x in d.groupby(by):
        z = zone.loc[x.index]
        N, K, n = len(x), int(x[flag].sum()), int(z.sum())
        obs += int((x[flag] & z).sum())
        exp += n * K / N if N else 0.0
        lo, hi = max(0, n - (N - K)), min(n, K)
        pmf = np.zeros(hi + 1)
        pmf[lo:] = hypergeom.pmf(np.arange(lo, hi + 1), N, K, n)
        dist = np.convolve(dist, pmf)
    return obs, exp, float(dist[obs:].sum()) if obs < len(dist) else 0.0


def wshare(d, mask):
    return float(d.loc[mask, "w"].sum() / d["w"].sum())


def standouts(g):
    cand = g.groupby("number")["nonm"].sum()
    cand = cand[cand >= 2]
    bar = 0.05 / len(cand)
    print(f"\n  standouts: {len(cand)} regions with two or more non-Muslims, each against the "
          f"rest in both halves ({' | '.join(','.join(h) for h in HALVES)}), bar P < {bar:.4f}:")
    found = set()
    for n in cand.index:
        ps = [stratified_p(d, d["number"] == n) for d in (g[g["wave"].isin(h)] for h in HALVES)]
        print(f"    {n:>2}  " + "   ".join(f"{o} vs {e:.1f} P={p:.3f}" for o, e, p in ps))
        if all(p < bar for _o, _e, p in ps):
            found.add(int(n))
    if found != STANDOUTS:
        raise SystemExit(f"regions standing apart: {sorted(found)}, not {sorted(STANDOUTS)}")


def afro_witness():
    a = afro.load(COUNTRY, expect_rounds=AFRO_ROUNDS, extra=["URBRUR"])
    a = a.rename(columns={"round": "wave"})
    muslim = a["category"].str.contains(AFRO_MUSLIM)
    other = sorted(set(a.loc[~muslim, "category"]) - AFRO_NON_MUSLIM)
    if other:
        raise SystemExit(f"Afrobarometer answers neither Muslim nor known non-Muslim: {other}")
    a["nonm"] = ~muslim
    a["urban"] = a["URBRUR"].astype(str).str.lower().eq("urban")
    if not a["URBRUR"].astype(str).str.lower().isin(["urban", "rural"]).all():
        raise SystemExit("Afrobarometer URBRUR has a value that is neither Urban nor Rural")
    lvl = pd.Series({r: wshare(x, x["nonm"]) for r, x in a.groupby("wave")})
    print(f"\n  Afrobarometer R5-R9 (witness): {int(a['nonm'].sum())} non-Muslims in {len(a):,}; "
          "weighted by round " + ", ".join(f"R{r} {v:.2%}" for r, v in lvl.items()))
    return a


def urban_test(g, a):
    d = g[g["urban"].notna()].copy()
    d["u"] = d["urban"].astype(bool)
    o1, e1, p1 = stratified_p(d, d["u"])
    o2, e2, p2 = stratified_p(a, a["urban"])
    print(f"\n  urban excess, exact within wave: Arab Barometer ({','.join(sorted(set(d['wave'])))}) "
          f"{o1} urban of {int(d['nonm'].sum())} against {e1:.1f} expected, P = {p1:.3f}; "
          f"Afrobarometer {o2} of {int(a['nonm'].sum())} against {e2:.1f}, P = {p2:.3f}")
    kn = d[d["nonm"] & d["u"]]
    cells = kn[kn["psu"].notna()].groupby(["wave", "psu"]).size()
    print(f"    largest PSU holds {int(cells.max())} of {len(kn)} urban non-Muslims "
          f"({cells.max() / len(kn):.0%}, cap {CELL_CAP:.0%})")
    passed = p1 < URBAN_BAR and p2 < URBAN_BAR and cells.max() / len(kn) <= CELL_CAP
    if passed != URBAN_EXPECTED:
        raise SystemExit(f"the urban test {'passes' if passed else 'fails'}, and the build was "
                         "written for the other outcome; read sources/ma.md §4 before changing it")
    ru = wshare(d[d["u"]], d.loc[d["u"], "nonm"])
    rr = wshare(d[~d["u"]], d.loc[~d["u"], "nonm"])
    print(f"    weighted: urban {ru:.3%}, rural {rr:.3%}, ratio {ru / rr:.2f}")
    return ru / rr


def foreign_mix():
    """{iso or region name: % of all foreigners}, checked against the study's own text."""
    mix = {}
    for region, (share, within) in MIX.items():
        w = dict(within)
        if region == "Sub-Saharan Africa":
            cm = 100 * CAMEROON_OF_ALL / share
            w["CM"], w["_other"] = cm, w["_other"] - cm
        # printed to one decimal: sub-Saharan Africa and the Maghreb add to 99.9
        if abs(sum(within.values()) - 100) > 0.15:
            raise SystemExit(f"{region}'s table does not add to 100")
        for k, v in w.items():
            mix[region if k == "_other" else k] = share * v / 100
    for iso, pct in TEXT_CHECK.items():
        if abs(mix[iso] - pct) > 0.1:
            raise SystemExit(f"{iso}: the tables give {mix[iso]:.2f}% and the text {pct}%")
    covered = sum(mix.values())
    print(f"\n  foreign mix: {len(mix)} rows cover {covered:.1f}% of foreigners; the study's "
          f"text agrees on {len(TEXT_CHECK)} named shares; the {100 - covered:.1f}% unnamed "
          "(Latin America, Oceania, stateless) is spread over the rest")
    return {k: v / covered for k, v in mix.items()}


def foreign_composition(mix):
    import origin_religion as origin

    with zipfile.ZipFile(PEW) as z:
        name = [n for n in z.namelist() if n.endswith("(percentages).csv")][0]
        pew = pd.read_csv(io.BytesIO(z.read(name)))
    pew = pew[pew["Year"] == 2020].set_index("Country")
    comp = {}
    for k, frac in mix.items():
        pn = origin.PEW_BY_ISO.get(k) if len(k) == 2 else f"All {k}"
        if pn not in pew.index:
            raise SystemExit(f"Pew has no row {pn!r} for {k}")
        row = {f: float(pew.loc[pn, f]) for f in origin.FAMILIES}
        for node, s in origin.composition(k if len(k) == 2 else "XX", row, "other.ma").items():
            node = "islam" if node.startswith("islam") else node
            comp[node] = comp.get(node, 0.0) + frac * s
    print("    foreign composition: " + ", ".join(f"{n} {v:.1%}" for n, v in
                                                  sorted(comp.items(), key=lambda x: -x[1])[:6]))
    return comp


def main():
    card()
    print("\n=== Arab Barometer, Morocco ===")
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
    sect = df["sect_m"].map(lambda v: ab.fold(v) if isinstance(v, str) else None)
    bad = df[(df["category"] != "Muslim") & sect.isin(SECT_ISLAM)]
    if len(bad):
        raise SystemExit(f"non-Muslim answers with a Muslim follow-up: {bad[['wave', 'category']]}")
    if (df["category"] == "Jewish").any():
        raise SystemExit("a Jewish answer is in the pool; read it before adding a category")

    g = df.copy()
    g["k"] = g["geo_raw"].map(key)
    bogus = g["k"].isin(BOGUS)
    if (g.loc[bogus, "category"] != "Muslim").any():
        raise SystemExit("a respondent with no region is not Muslim")
    print(f"  {int(bogus.sum())} respondents with no region (all Muslim) leave the geography")
    g = g[~bogus].copy()
    g["number"] = g["k"].map(NORM)
    if g["number"].isna().any():
        raise SystemExit(f"Q1 labels with no region: {sorted(set(g.loc[g['number'].isna(), 'geo_raw']))}")
    g["number"] = g["number"].astype(int)
    for w, base in CODE_SYSTEMS.items():
        sub = g[g["wave"] == w]
        nbad = int(((sub["geo_code"] - base) != sub["number"]).sum())
        if nbad:
            raise SystemExit(f"wave {w}: {nbad} Q1 codes disagree with the region name")
    print(f"  {g['geo_raw'].nunique()} Q1 labels -> 12 regions; every code agrees with its name")
    g["urban"] = g["urban"].map(lambda v: None if not isinstance(v, str) else v.lower() == "urban")

    pc = pd.read_csv(PIECES)
    reg_pop = pc.groupby("region")[["mor_urban", "mor_rural"]].sum().sum(axis=1)
    g["geo_id"] = g["number"].map(lambda n: f"R{n:02d}")
    ab.held_out(g, reg_pop.rename(lambda n: f"R{n:02d}"), COUNTRY, pop_source="RGPH 2024 Moroccans")
    ab.assert_not_quota(g, COUNTRY)

    g["nonm"] = g["category"] != "Muslim"
    standouts(g)
    a = afro_witness()
    k = urban_test(g, a)

    L = wshare(g, g["nonm"])
    print(f"\n  weighted non-Muslim share, waves V-VIII: {L:.3%}; by wave "
          + ", ".join(f"{w} {wshare(x, x['nonm']):.2%}" for w, x in g.groupby('wave', sort=False)))
    if not NONMUSLIM_BAND[0] <= L <= NONMUSLIM_BAND[1]:
        raise SystemExit(f"{L:.3%} is outside {NONMUSLIM_BAND}")
    u = pc["mor_urban"].sum() / (pc["mor_urban"].sum() + pc["mor_rural"].sum())
    s_r = L / (u * k + 1 - u)
    s_u = k * s_r
    print(f"    Moroccans {u:.1%} urban (RGPH 2024): drawn urban {s_u:.3%}, rural {s_r:.3%}")
    nm = g[g["nonm"]]
    comp = (nm.groupby("category")["w"].sum() / nm["w"].sum()).reindex(NON_MUSLIM, fill_value=0.0)
    print("    non-Muslim mix (weighted): " + ", ".join(f"{c} {v:.1%} (n={int((nm['category'] == c).sum())})"
                                                  for c, v in comp.items()))

    got = dict(pooled=pooled, named=named, respondents=len(g), nonmuslim=len(nm),
               christian=int((nm["category"] == "Christian").sum()),
               no_religion=int((nm["category"] == "No religion").sum()),
               other=int((nm["category"] == "Other").sum()))
    print(f"\n  note_public's survey figures: {got}")
    if NOTE and got != NOTE:
        raise SystemExit(f"note_public's figures are {NOTE}; the pool gives {got}")

    # ---- Moroccans, per piece then per unit ----
    lut = pd.read_csv(LOOKUP)
    names = dict(zip(lut["geo_id"], lut["name"]))
    pc["mor"] = pc["mor_urban"] + pc["mor_rural"]
    pc["nonm"] = pc["mor_urban"] * s_u + pc["mor_rural"] * s_r
    per = pc.groupby("geo_id")[["mor", "nonm"]].sum()
    m = pd.DataFrame({"Muslim": per["mor"] - per["nonm"],
                      **{c: per["nonm"] * comp[c] for c in NON_MUSLIM}})[CATS]
    counts = round_within_rows(m)
    if not (counts.sum(axis=1) == per["mor"]).all():
        raise SystemExit("a unit's rounded counts do not sum to its Moroccans")
    out = counts.stack().rename("count").reset_index()
    out.columns = ["geo_id", "source_category", "count"]
    out["geo_level"] = "province"
    out["geo_name"] = out["geo_id"].map(names)
    out["basis"] = "self_id"
    out["year"] = YEARS
    out["source_id"] = SOURCE_ID
    out["note"] = ("Arab Barometer waves V to VIII pooled; the national non-Muslim share split "
                   "urban and rural, national mix, on the RGPH 2024 count of Moroccans")
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
            "source_id", "note"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8")

    # ---- foreign residents ----
    fcomp = foreign_composition(foreign_mix())
    pc["for"] = pc["for_urban"] + pc["for_rural"]
    fper = pc.groupby("geo_id")["for"].sum()
    nodes = sorted(fcomp)
    fm = pd.DataFrame({n: fper * fcomp[n] for n in nodes})
    fcounts = round_within_rows(fm)
    if not (fcounts.sum(axis=1) == fper).all():
        raise SystemExit("a unit's rounded foreign counts do not sum to its foreigners")
    ext = fcounts.stack().rename("count").reset_index()
    ext.columns = ["geo_id", "node", "count"]
    ext = ext[ext["count"] > 0]
    ext["geo_level"] = "province"
    ext["geo_name"] = ext["geo_id"].map(names)
    ext["tier"] = "modelled"
    ext["basis"] = "nationality_derived"
    ext["year"] = 2024
    ext["source_id"] = "rgph2024_foreigners_x_pew2020"
    ext[["geo_id", "geo_level", "geo_name", "node", "count", "tier", "basis", "year",
         "source_id"]].to_csv(OUT_FOREIGN, index=False, encoding="utf-8")

    total = int(out["count"].sum()) + int(ext["count"].sum())
    print(f"\nwrote {OUT} ({int(out['count'].sum()):,} Moroccans) and {OUT_FOREIGN} "
          f"({int(ext['count'].sum()):,} foreigners, {ext['node'].nunique()} nodes); {total:,} people")
    drawn = out.groupby("source_category")["count"].sum()
    for c in CATS:
        print(f"    {drawn[c] / total:8.3%}  {c}  ({int(drawn[c]):,})")
    fx = ext.groupby("node")["count"].sum().sort_values(ascending=False)
    print("    foreign: " + ", ".join(f"{n} {int(v):,}" for n, v in fx.head(6).items()))


if __name__ == "__main__":
    main()
