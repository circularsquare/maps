"""Arab Barometer — the parts any country drawn from it shares.

`sources.md` §11af is the assessment of the source; `sources/eg.md` is the first country built
on it and carries the worked argument. This is the twin of `sources/lapop.py`, deliberately
so: the construction is the same one (pool the waves that asked, weight, cut by the survey's
own subnational unit, apply the shares to a population table somebody else counted), and
§11af found four more countries east of Egypt that it could serve. Nine hand-copied versions
of it would drift apart in exactly the places that matter.

What is here is only what is genuinely the same in every country. What is NOT here, and must
stay in each `sources/<cc>.py`:

  * **the decode from the survey's `Q1` label to a polygon.** In this survey that is not a
    code join at all, and the reason is the next section.
  * `CARRIES`, the categories drawn on their own unit shares, asserted per country.
  * `OVERRIDE`, where a country draws a category that FAILED the split-half. One named
    category, a written reason, and a person's decision.
  * every word of the docstring a reader will actually consult.

## THE NUMERIC CODES ARE NOT THE SAME QUESTION TWICE, AND POOLING ON THEM IS WRONG

`Q1012` is *"What is your religion?"* in every wave and its **value codes are re-used for
different answers between waves**:

    wave III   1 Muslim  2 Christian  3 Other   4 Jewish (Yemen only)
    wave IV    1 Muslim  2 Christian  3 Other   4 Jewish
    wave V     1 Muslim  2 Christian  3 JEWISH  4 ATHEIST      5 other
    wave VII   1 Muslim  2 Christian  3 Other   4 NO RELIGION

So code 3 is `Other` in three waves and `Jewish` in the fourth, and code 4 is `Jewish`,
`Atheist` and `No religion` depending on which year you are in. A pooled frame keyed on the
code silently merges three different answers, and nothing about the result looks wrong —
`[[reference_pooled_survey_labels]]` on the religion column rather than on the geography.
**`load()` therefore decodes every wave through that wave's OWN label set before anything is
pooled, and the label string is the category from then on.** The same applies to `Q1`, where
the same file gives one country 45 spellings of 27 units.

**And the answer card is not the same card twice either**, which the decode cannot fix and
the country module has to decide about. Only wave V offers `Atheist`; only wave VII offers
`No religion`; waves III and IV offer neither. A share pooled across all four for an option
that was on one of the four cards is measuring which questionnaire was used. `sources/eg.py`
has the worked case.

## The construction, in one paragraph

Pool the waves that asked, weight by `wt`, cut by `Q1`. That gives a share per unit. The
magnitude comes from a population table the country module chooses and joins. **No magnitude
is invented: every person drawn is a person that table counts in that unit, and the survey
only decides the column** (spec §14.4 rule 1). Every row is `modelled` in §7's sense, because
nobody counted religion in any of these countries.

## Access

`arabbarometer.org/survey-data/data-downloads/` renders every download as `href="#"` behind a
name/email form, which reads as a wall. It is not one: the real file URLs are in the page's
own HTML and are the ones below. Arab Barometer's FAQ: *"Anyone can download the Arab
Barometer data for analysis at no cost"*, the data are *"publicly available and free of
charge"*, and the form is a request step rather than a licence. No retrieval restriction, no
redistribution clause, no stated citation requirement. Read before anything was downloaded,
because of Nişanyan (§11ac).
"""

import os
import ssl
import sys
import urllib.request
import warnings
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AB_DIR = os.path.join(ROOT, "data", "raw", "arabbarometer")

BASE = "https://www.arabbarometer.org/wp-content/uploads/"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")

# (wave, ordinal, zip, the .sav inside it). The ordinal is what the split-half sorts on:
# roman numerals do not sort and "VII" < "V" as a string.
WAVES = [
    ("I",    1, "ABI_English.sav_.zip",                  "ABI_English.sav"),
    ("II",   2, "ABII_English.sav_.zip",                 "ABII_English.sav"),
    ("III",  3, "ABIII_English.sav_.zip",                "ABIII_English.sav"),
    ("IV",   4, "ABIV_English.sav_.zip",                 "ABIV_English_Updated.sav"),
    ("V",    5, "ArabBarometer_WaveV_ENG.zip",           "ArabBarometer_WaveV_English_v2.sav"),
    ("VI-1", 6, "ENG-Arab-Barometer-Wave-VI-Part-1_DEC.zip",
     "Arab_Barometer_Wave_6_Part_1_ENG_RELEASE.sav"),
    ("VI-2", 6, "ENG-Arab-Barometer-Wave-VI-Part-2-1.zip",
     "Arab_Barometer_Wave_6_Part_2_ENG_RELEASE.sav"),
    ("VI-3", 6, "ENG-Arab-Barometer-Wave-VI-Part-3_DEC.zip",
     "Arab_Barometer_Wave_6_Part_3_ENG_RELEASE.sav"),
    ("VII",  7, "AB7_English_Version6.zip",              "AB7_ENG_Release_Version6.sav"),
    ("VIII", 8, "ArabBarometer_WaveVIII_English_v2.zip",
     "ArabBarometer_WaveVIII_English_v3.sav"),
]

# A category is ELIGIBLE for its own geography if it is at least this much of the country.
# The floor is `sources/lapop.py`'s and the reasoning is §11ad's: that instrument's
# sub-national cut was measured failing below about 1% against two censuses. Eligibility is
# not permission — `stability()` is the stricter test and it decides.
ELIGIBLE_FLOOR = 0.01


def _ctx():
    ctx = ssl.create_default_context()
    # arabbarometer.org's chain has been seen to fail verification from here while serving
    # the right bytes; the files are public data and are checked by zip magic below.
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def fetch(waves=None):
    """Download and unzip the waves named, or all of them. ~46 MB of zips."""
    os.makedirs(AB_DIR, exist_ok=True)
    want = set(waves) if waves else {w[0] for w in WAVES}
    for name, _ord, zipname, savname in WAVES:
        if name not in want:
            continue
        dst = os.path.join(AB_DIR, zipname)
        if not os.path.exists(dst) or os.path.getsize(dst) < 100_000:
            req = urllib.request.Request(BASE + zipname,
                                         headers={"User-Agent": UA, "Referer": BASE})
            with urllib.request.urlopen(req, timeout=900, context=_ctx()) as r:
                data = r.read()
            # §5a: a 200 is not a download.
            if data[:2] != b"PK":
                raise SystemExit(f"{zipname} is not a zip — starts {data[:16]!r}")
            with open(dst + ".part", "wb") as f:
                f.write(data)
            os.replace(dst + ".part", dst)
            print(f"  got  {zipname} ({os.path.getsize(dst):,} bytes)")
        else:
            print(f"  have {zipname} ({os.path.getsize(dst):,} bytes)")
        unzip(zipname)
        if not os.path.exists(os.path.join(AB_DIR, savname)):
            raise SystemExit(f"{zipname} does not contain {savname} — AB has re-released it")


def unzip(zipname=None):
    """Extract every .sav out of the zips on disk, ignoring the macOS resource forks."""
    names = [zipname] if zipname else [f for f in os.listdir(AB_DIR) if f.endswith(".zip")]
    for fn in names:
        with zipfile.ZipFile(os.path.join(AB_DIR, fn)) as z:
            for i in z.infolist():
                if i.filename.endswith(".sav") and "__MACOSX" not in i.filename:
                    out = os.path.join(AB_DIR, os.path.basename(i.filename))
                    if not os.path.exists(out):
                        with z.open(i) as s, open(out, "wb") as d:
                            d.write(s.read())


def _col(df, *names):
    up = {c.upper(): c for c in df.columns}
    for n in names:
        if n.upper() in up:
            return up[n.upper()]
    return None


def load(country, expect_waves=None):
    """One country's respondents, decoded wave by wave through that wave's own labels.

    Returns [`wave`, `wave_no`, `category`, `geo_raw`, `geo_code`, `w`], one row per
    respondent who answered `Q1012`. `category` is the ANSWER'S OWN WORDING (spec §2.4) and
    never a numeric code; see the module docstring for why that is not a style preference.

    `expect_waves` is the list of wave names the country is known to appear in, asserted so a
    re-release that adds or drops a wave is a failure here rather than a quiet re-levelling.
    """
    import pyreadstat

    frames = []
    for name, ordinal, _zipname, savname in WAVES:
        p = os.path.join(AB_DIR, savname)
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing — run the country module with --fetch")
        df, meta = pyreadstat.read_sav(p)
        c = _col(df, "country")
        rel = _col(df, "q1012")
        geo = _col(df, "q1")
        wt = _col(df, "wt", "weight", "weight1500")
        if c is None:
            raise SystemExit(f"wave {name} has no country column")
        clab = meta.variable_value_labels.get(c, {})
        cname = df[c].map(clab).fillna(df[c].astype(str)).astype(str).str.strip()
        sub = df[cname.str.lower() == country.lower()]
        if not len(sub):
            continue
        if rel is None:
            raise SystemExit(f"wave {name} has {country} rows but no Q1012")
        rlab = meta.variable_value_labels.get(rel, {})
        glab = meta.variable_value_labels.get(geo, {}) if geo else {}
        undecoded = sorted(set(sub[rel].dropna()) - set(rlab))
        if undecoded:
            raise SystemExit(f"wave {name}: Q1012 codes with no label: {undecoded}")
        frames.append(pd.DataFrame({
            "wave": name,
            "wave_no": ordinal,
            "category": sub[rel].map(rlab),
            "geo_code": sub[geo] if geo is not None else np.nan,
            "geo_raw": (sub[geo].map(glab) if geo is not None else np.nan),
            "w": pd.to_numeric(sub[wt], errors="coerce") if wt else 1.0,
        }))
        print(f"  wave {name:<5} n={len(sub):>6}  Q1012 answered by "
              f"{int(sub[rel].notna().sum()):>6}  "
              f"{'' if geo is None else str(sub[geo].map(glab).nunique()) + ' Q1 labels'}")
    if not frames:
        raise SystemExit(f"no {country} rows in any wave")
    out = pd.concat(frames, ignore_index=True)
    out = out[out["category"].notna()].copy()
    out["w"] = out["w"].fillna(1.0)

    waves = [w for w in dict.fromkeys(n for n, _o, _z, _s in WAVES) if w in set(out["wave"])]
    if expect_waves is not None and waves != list(expect_waves):
        raise SystemExit(f"{country} now appears in waves {waves}, expected "
                         f"{list(expect_waves)} — a re-release has changed the pool")
    return out


def national(df, cat_col="category"):
    """Weighted national share per answer, pooled over the waves in `df`."""
    return df.groupby(cat_col)["w"].sum() / df["w"].sum()


def held_out(df, pop, country, unit_col="geo_id", n_perm=20000, seed=0,
             pop_source="the population table"):
    """Test the unit decode without touching the religion column.

    THE TEST IS THE PERMUTATION, NOT THE CORRELATION. The survey's weighted share of
    respondents per unit is compared with the population table's share, and what makes it
    evidence is how that compares with the wrong answers: the unit labels are shuffled
    `n_perm` times and the observed r is ranked against those. A correlation any random
    pairing could produce says nothing; one that beats every random pairing pins the decode.
    `sources/lapop.py` has the longer argument and `sources/kz.py` the original.

    A survey that quota-samples by governorate will do very well here and that is fine — the
    question this answers is whether the NAMES were joined to the right polygons, which a
    permutation is exactly the test for.
    """
    print("\n  held-out check (nothing here touches the religion column):")
    share_s = df.groupby(unit_col)["w"].sum() / df["w"].sum()
    share_p = pop / pop.sum()
    j = pd.concat([share_s.rename("survey"), share_p.rename("pop")], axis=1).dropna()
    if len(j) != len(share_s):
        raise SystemExit(f"{len(share_s) - len(j)} sampled units have no population")
    r = np.corrcoef(j["survey"], j["pop"])[0, 1]
    ratio = (j["survey"] / j["pop"]).sort_values()
    print(f"    unit share of respondents vs {pop_source}:  r = {r:+.3f} over {len(j)} units")
    print(f"      thinnest sampled {ratio.index[0]} at {ratio.iloc[0]:.2f}x its population "
          f"share, fullest {ratio.index[-1]} at {ratio.iloc[-1]:.2f}x")

    rng = np.random.default_rng(seed)
    a, b = j["survey"].to_numpy(), j["pop"].to_numpy()
    perm = np.array([np.corrcoef(a, rng.permutation(b))[0, 1] for _ in range(n_perm)])
    beaten = int((perm >= r).sum())
    print(f"      against {n_perm:,} random pairings of the same units: best random "
          f"r = {perm.max():+.3f}, and {beaten} reach the observed one")
    if beaten:
        raise SystemExit(
            f"{beaten} of {n_perm} random pairings of {country}'s units match or beat the "
            f"observed r={r:+.3f}. The population check does not pin this decode, so the "
            "join needs a witness that does before anything is drawn.")
    return r


def stability(df, nat, n_units, unit_col="geo_id", cat_col="category", override=None):
    """WHICH CATEGORIES CARRY THEIR OWN GEOGRAPHY — the split-half decides it.

    Spec §14.16's test. Each category is ranked across the units in the earlier half of the
    waves and again in the later half, and the two orderings are compared. A category whose
    ranking does not replicate has not demonstrated that it HAS a geography, whatever its
    pooled spread looks like. The bar is 1.96/sqrt(n-1), which is what a Spearman correlation
    over `n` units needs to be distinguishable from zero at 95%; it is never moved to make
    something pass, because moving it would silently change every other category too.

    **HARMONISE THE UNITS BEFORE CALLING THIS.** §11af ran it on Egypt's raw `Q1` labels and
    got +0.416 against a +0.566 bar, a clear failure — because only 13 of 27 units appeared in
    both halves and the test was being run on mangled units. Harmonised, the same data give
    +0.495 against +0.418 and pass. A stability test run before the units exist reports noise
    as a negative, and a negative is what this project treats as evidence.

    `override` is `{category: "the reason"}` and is a person's decision, not an agent's; see
    `sources/lapop.py`'s docstring for when it is even available.
    """
    override = override or {}
    bar = 1.96 / np.sqrt(n_units - 1)
    waves = sorted(df["wave_no"].unique())
    cut = waves[len(waves) // 2]
    early, late = df[df["wave_no"] < cut], df[df["wave_no"] >= cut]
    print(f"\n  split-half stability across waves (§14.16), bar = +{bar:.3f} at 95% on "
          f"{n_units} units:")
    print(f"    waves {sorted(set(early['wave']))} n={len(early):,}   "
          f"waves {sorted(set(late['wave']))} n={len(late):,}")
    print(f"    {'answer':<28}{'national':>10}{'spearman':>10}{'pearson':>9}  verdict")

    carries = []
    for c in sorted(nat.index, key=lambda k: -nat[k]):
        if nat[c] < ELIGIBLE_FLOOR:
            print(f"    {str(c)[:26]:<28}{nat[c] * 100:9.2f}%{'':>10}{'':>9}  "
                  f"too small to place (§11ad)")
            continue

        def share(d):
            return d.groupby(unit_col).apply(
                lambda x: x.loc[x[cat_col] == c, "w"].sum() / x["w"].sum(),
                include_groups=False)

        j = pd.concat([share(early).rename("e"), share(late).rename("l")], axis=1).dropna()
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp = j["e"].corr(j["l"], method="spearman")
            pe = j["e"].corr(j["l"])
        passed = bool(np.isfinite(sp)) and sp >= bar
        forced = c in override
        if passed or forced:
            carries.append(c)
        shown = f"{sp:+10.3f}{pe:+9.3f}" if np.isfinite(sp) else f"{'undefined':>10}{'':>9}"
        if passed:
            verdict = f"own geography  ({len(j)} units in both halves)"
        elif forced:
            verdict = "own geography — UNDER THE BAR, drawn on Anita's call"
        else:
            verdict = "NOT distinguishable from zero"
        print(f"    {str(c)[:26]:<28}{nat[c] * 100:9.2f}%{shown}  {verdict}")
        if forced and not passed:
            print(f"        reason: {override[c]}")
    stale = sorted(set(override) - set(nat.index))
    if stale:
        raise SystemExit(f"override names categories this country does not have: {stale}")
    return carries


def build(df, nat, large, small, pop, units, unit_col="geo_id", cat_col="category",
          unit_noun="unit"):
    """Shares x population -> counts, as a closed partition of every unit.

    THE MEASURED SHARES PASS THROUGH UNTOUCHED. A unit's share of a category that cleared the
    split-half is applied as it stands; what is left of that unit is divided among the rest at
    their NATIONAL relative proportions, so the tail's geography is the residual of the stable
    measurements rather than a flat national rate. `sources/lapop.py` has the argument.

    **`small` MAY BE EMPTY**, which lapop's version could not express: a country whose answer
    card has two boxes and both clear the bar has no tail at all, and its unit shares are a
    closed partition already. Egypt is that country. The residual is then exactly zero and
    must not be treated as "no room for the tail".
    """
    by_unit = df.groupby([unit_col, cat_col])["w"].sum().unstack(fill_value=0.0)
    for c in nat.index:
        if c not in by_unit.columns:
            by_unit[c] = 0.0
    unit_share = by_unit.div(by_unit.sum(axis=1), axis=0)

    residual = 1.0 - unit_share[large].sum(axis=1)
    small_total = float(sum(nat[c] for c in small))
    if small:
        if (residual <= 0).any():
            raise SystemExit(f"units with no room for the tail: "
                             f"{sorted(residual[residual <= 0].index)}")
        print(f"    the tail is {residual.min():.1%} of {residual.idxmin()} and "
              f"{residual.max():.1%} of {residual.idxmax()}, against {small_total:.1%} "
              "nationally")
    else:
        if residual.abs().max() > 1e-9:
            raise SystemExit("no tail categories, but the large ones do not sum to 1 in "
                             f"every unit (worst {residual.abs().max():.3g})")
        print(f"    no tail: the {len(large)} drawn answers are a closed partition of every "
              f"{unit_noun}")

    rows = []
    for unit in units:
        p = int(pop[unit])
        for c in large:
            rows.append((unit, c, unit_share.loc[unit, c] * p, f"{unit_noun} share"))
        for c in small:
            rows.append((unit, c, residual[unit] * (nat[c] / small_total) * p,
                         f"national share within the {unit_noun}'s residual"))

    out = pd.DataFrame(rows, columns=["geo_id", "source_category", "count", "basis_note"])
    out["count"] = out["count"].round().astype("int64")
    drift = int(sum(int(pop[u]) for u in units)) - int(out["count"].sum())
    if abs(drift) > len(out):
        raise SystemExit(f"rounding drift {drift} is larger than one person per row")
    if drift:
        out.loc[out["count"].idxmax(), "count"] += drift
    print(f"  rounding drift {drift:+d} people, absorbed into the largest cell")
    return out
