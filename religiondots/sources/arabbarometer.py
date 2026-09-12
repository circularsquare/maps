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

**And the label string is then guarded in its turn**, because the same failure exists one
column across: pooling on a raw wording splits an answer that two waves spell differently.
`assert_one_wording()` runs at the foot of `load()` and refuses to return a pool in which one
answer arrives under two spellings. Lebanon is the measured case — `Other` in three waves,
`other` in a fourth — and that function's docstring is where it is written up.

**And the answer card is not the same card twice either**, which the decode cannot fix and
the country module has to decide about. Only wave V offers `Atheist`; only wave VII offers
`No religion`; waves III and IV offer neither. A share pooled across all four for an option
that was on one of the four cards is measuring which questionnaire was used. `sources/eg.py`
has the worked case.

## AND THE FIELDWORK CAN SET THE ANSWER, WHICH IS A DIFFERENT FAILURE FROM ALL THREE ABOVE

The three guards above are about a pooled file spelling one thing two ways. This one is about
the file being right and the sample being built to a specification. **Arab Barometer's Lebanese
sample is a fixed sect-by-governorate quota** — waves V and VII return the same Christian count
in all eight governorates three years apart, and Kesrwan-Jbeil comes back 100% Christian in all
three parts of wave VI. Nothing in the frame is wrong; the composition is simply not a
measurement of Lebanon.

**`stability` cannot see it, and passes it at its strongest**, because a quota replicates
between the wave halves by construction. So `assert_not_quota` runs first, `quota_agreement` is
the statistic, and Lebanon is not drawn: `sources/lb.md` and `sources.md` §11al. Jordan, Egypt
and Iraq were measured against it and are clean.

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

import itertools
import math
import os
import re
import ssl
import sys
import unicodedata
import urllib.request
import warnings
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import spearman_null

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

WAVE_ORDER = {name: i for i, (name, _o, _z, _s) in enumerate(WAVES)}
WAVE_NAMES = [name for name, _o, _z, _s in WAVES]

# Below this many possible orderings of a country's units, `held_out` checks every one of them
# instead of sampling. 8! = 40,320 is under it and 9! = 362,880 is over. See `held_out`.
EXACT_PERM_MAX = 50_000

# A category is ELIGIBLE for its own geography if it is at least this much of the country.
# The floor is `sources/lapop.py`'s and the reasoning is §11ad's: that instrument's
# sub-national cut was measured failing below about 1% against two censuses. Eligibility is
# not permission — `stability()` is the stricter test and it decides.
ELIGIBLE_FLOOR = 0.01

# THE BAR ON `assert_not_quota`, PRE-REGISTERED AND NOT TO BE TUNED. The smallest pairwise
# agreement p-value, Bonferroni-multiplied by the number of wave pairs compared, must exceed
# this. Measured 2026-09-09 on the four countries this file has been read for: **Lebanon
# 1.2e-4** (wave V against wave VII, four free cells and four exact agreements), against
# **Jordan 21, Egypt 7.2 and Iraq 3.0** — the three of those are Bonferroni products above 1,
# i.e. nothing at all. There is no country anywhere near the bar; it separates a quota from a
# survey by four orders of magnitude. `quota_agreement`'s docstring is the write-up.
QUOTA_P_BAR = 1e-3


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


# `8. Jordan`, `5. Egypt`, `17. Saudi Arabia` — wave II, and only wave II, prints the country's
# position in its own code list inside the label. See `country_key`.
_ORDINAL_PREFIX = re.compile(r"^\s*\d+\s*[.)]\s*")

# The `country` column is a per-wave label set exactly as the answers are, so a country spelled
# two ways loses whole waves without failing anything. `country_key` folds the one difference
# that is mechanical (wave II's ordinal) and stops; this names the differences that are not.
# A country named here is matched on ALL of its spellings, and the key may be any of them.
#
# **Saudi Arabia is the only case in waves I to VIII**, swept 2026-09-09 over every `country`
# value label declared in all ten files: wave II spells it `17. Saudi Arabia` and wave V's
# label set spells it `Kingdom of Saudi Arabia`, so after the ordinal strip those are two keys
# and asking for either one finds half the file. `assert_no_near_miss` is what found it and is
# what will find the next one; this dictionary is where the answer gets written down.
#
# It buys nothing for Saudi Arabia itself, and that is worth knowing before anyone tries:
# **Saudi Arabia cannot be drawn from this survey at all.** Its 1,404 wave II respondents have
# a governorate and a weight and an entirely empty `Q1012`, and wave V declares the label
# against zero rows. The alias is here so that the next country with two spellings is a line
# of dictionary rather than a silent half-pool.
COUNTRY_ALIASES = {
    "saudi arabia": ["Saudi Arabia", "Kingdom of Saudi Arabia"],
}

# Words that name a form of state, or join two words that do, and never name a place.
# `near_miss_keys` uses them to tell `Kingdom of Saudi Arabia` against `Saudi Arabia`, which is
# one country under two labels, from `South Sudan` against `Sudan`, which is two countries: the
# extra words in the first pair are all in here and `south` is not. Widening this asserts that
# two more labels are one country, so it is a written decision and not a convenience.
_FORM_OF_STATE = {
    "and", "arab", "democratic", "federal", "federation", "great", "hashemite", "islamic",
    "jamahiriya", "kingdom", "of", "people", "peoples", "people's", "popular", "republic",
    "socialist", "state", "states", "sultanate", "the", "union", "united",
}

# WAVES A COUNTRY LEAVES OUT OF ITS POOL ALTHOUGH THE FILES OFFER THEM, AND WHY.
#
# `load` refuses a pool that is quietly narrower than the files, because that is the exact
# shape of the wave II failure `country_key` writes up: a wave present in the file, absent from
# the pool, and nothing on screen. Leaving a wave out is often right; leaving it out in silence
# never is. A country module normally states its own with `omit=` at the call.
#
# Egypt's is here instead of in `sources/eg.py`, for one reason. It is a decision about an
# ALREADY-DRAWN country and the pin is Anita's to rule on, so the country module is not to be
# edited while that is open; `sources/jo.md` §9.3 costs the rebuild out. Move this into
# `sources/eg.py` beside its `WAVES` whenever the ruling lands.
OMITTED = {
    "egypt": {
        "II": "Egypt was drawn on 2026-09-08 from a pool that could not see wave II, and "
              "fixing the country filter is not a reason to move a published map. Adding the "
              "wave takes the national Christian share from 6.024% to 5.831% and puts Asyut "
              "above Minya as the most Christian governorate on the strength of 70 "
              "interviews; sources/jo.md §9.3 has the measurement. Anita's to rule on.",
    },
}


def country_key(label):
    """Fold a `country` value label to something two waves' spellings of it share.

    ## WAVE II WAS SILENTLY MISSING FROM EVERY COUNTRY BUILT HERE UNTIL 2026-09-08

    `load()` selected a country by `label.lower() == country.lower()`, which is right for nine
    of the ten waves and wrong for wave II, whose `country` value labels read **`8. Jordan`,
    `5. Egypt`, `17. Saudi Arabia`** — the numeric code repeated inside the label. So the
    comparison failed for every country, `load()` found no rows, and the wave was skipped by
    the `continue` that is there for a country a wave did not field. **No error, no warning,
    and a smaller pool than the file contains**: 1,188 Jordanians and 1,219 Egyptians, with
    `q1012 Religion` and `q1 Province/Governorate/State` both present and a weight.

    That is `[[reference_pooled_survey_labels]]` on a THIRD column of this same survey. The
    module docstring has it on the religion answers, `assert_one_wording` has it on their
    wordings, and this is it on the country itself. All three are the same mechanism: a pooled
    file keyed on a label that one wave spells its own way.

    Stripping the ordinal is deliberately the ONLY thing done here. `casefold` and a strip
    handle wave I's lower-cased names; nothing else is folded, because two Arab Barometer
    countries with similar names are a real possibility and a fuzzy country key would be the
    worst possible place to be clever. **`COUNTRY_ALIASES` is where the differences that are
    not mechanical get written down**, and `near_miss_keys` is what finds them: the same
    stripping leaves wave II's `17. Saudi Arabia` and wave V's `Kingdom of Saudi Arabia` as
    two keys, which is this failure again one release later.

    **A country module that does not want wave II must say so with `waves=` AND `omit=`**,
    which is what `OMITTED` does for `sources/eg.py` and why Egypt's numbers did not move when
    this was fixed. `waves=` alone is no longer enough: a pool narrower than the files without
    a stated reason is the failure this docstring is about.
    """
    return _ORDINAL_PREFIX.sub("", str(label)).strip().casefold()


def country_spellings(country):
    """Every `country` label this country may arrive under, as `country_key`s.

    One spelling is the normal case. The lookup is symmetric, so a module that asks for
    `Kingdom of Saudi Arabia` gets the same waves as one that asks for `Saudi Arabia`.
    """
    want = country_key(country)
    for k, group in COUNTRY_ALIASES.items():
        keys = {country_key(n) for n in list(group) + [k]}
        if want in keys:
            return keys
    return {want}


def near_miss_keys(key, others):
    """The keys in `others` that are the same country name as `key` under a form-of-state word.

    ONE COUNTRY UNDER TWO LABELS IS THE THIRD PLACE THIS SURVEY LOSES A WAVE IN SILENCE. The
    module docstring has the first (a code re-used for a different answer), `assert_one_wording`
    the second (an answer spelled two ways), and `country_key` the third, which it fixes only
    as far as wave II's ordinal. It stops there on purpose: two Arab Barometer countries with
    similar names are a real possibility and a fuzzy country key would be the worst place in
    this module to be clever.

    So this DETECTS rather than resolves, the same division of labour as `assert_one_wording`.
    Two keys are a near miss when the words of one are a strict subset of the words of the
    other and **every extra word is a form of state**: `saudi arabia` inside `kingdom of saudi
    arabia`, `egypt` inside `arab republic of egypt`, `emirates` inside `united arab emirates`.
    `sudan` inside `south sudan` is not one, and must not be, because those are two countries.
    """
    mine = set(str(key).split())
    hits = []
    for other in others:
        theirs = set(str(other).split())
        if mine == theirs:
            continue
        small, big = (mine, theirs) if len(mine) < len(theirs) else (theirs, mine)
        if small < big and (big - small) <= _FORM_OF_STATE:
            hits.append(other)
    return hits


def assert_no_near_miss(country, declared):
    """Refuse to pool a country whose name also appears in this file under a second spelling.

    `declared` is `wave_coverage`'s second return: every `country` value label declared
    anywhere in the ten files, folded, against the waves declaring it. The check is against
    the DECLARED labels rather than the observed ones, because a label with no rows in the
    wave you looked at is precisely the label that has rows in the wave you did not; wave V
    declares `Kingdom of Saudi Arabia` over zero rows, and that is the only reason the case
    was findable before somebody built the country on half its waves.
    """
    want = country_spellings(country)
    hits = sorted({o for k in want for o in near_miss_keys(k, [d for d in declared
                                                             if d not in want])})
    if not hits:
        return
    lines = [f"    {k!r} is declared in waves "
             f"{', '.join(declared.get(k, [])) or '(none)'}" for k in sorted(want)]
    lines += [f"    {o!r} is declared in waves {', '.join(declared[o])}" for o in hits]
    raise SystemExit(
        f"{country} resolves to {sorted(want)}, and this file declares another country label "
        "that is the same name under a form-of-state word:\n" + "\n".join(lines)
        + "\n  Either those are one country spelled two ways, in which case the pool you are "
        "about to build is missing the waves on the other line and nothing downstream would "
        "have said so, or they are two countries whose names nest. This module cannot tell "
        "which and does not guess (see `country_key`). If they are one country, name every "
        "spelling in COUNTRY_ALIASES; if they are two, `_FORM_OF_STATE` is what has to "
        "change, with the reason written beside it.")


def wave_coverage(country):
    """WHAT THE FILES HOLD FOR THIS COUNTRY, read before anything is pooled.

    Returns `(available, declared)`:

      * **`available`** — the waves, in file order, holding at least one respondent of this
        country with a non-null `Q1012`. This is what `expect_waves` is asserted against, and
        reading it here rather than off the pooled frame is the whole point of the function.
        `load` computed that assertion from a frame it had ALREADY filtered by `waves=`, so a
        wave the pool never asked for could not appear in it, and the guard whose docstring
        promises to catch a re-release adding a wave was blind to exactly that. An assertion
        that cannot fail is worse than none.
      * **`declared`** — every `country` value label declared in any wave, folded by
        `country_key`, against the waves declaring it. Declared is not present: wave V declares
        `Kingdom of Saudi Arabia` and holds no rows under it. `assert_no_near_miss` reads this.

    It costs about a second and a half over all ten waves, because it reads the metadata on its
    own and then two columns of the data, never the whole file.
    """
    import pyreadstat

    want = country_spellings(country)
    available, declared = [], {}
    for name, _ordinal, _zipname, savname in WAVES:
        p = os.path.join(AB_DIR, savname)
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing — run the country module with --fetch")
        meta = pyreadstat.read_sav(p, metadataonly=True)[1]
        names = pd.DataFrame(columns=list(meta.column_names))
        c, rel = _col(names, "country"), _col(names, "q1012")
        if c is None:
            raise SystemExit(f"wave {name} has no country column")
        clab = meta.variable_value_labels.get(c, {})
        for lab in clab.values():
            declared.setdefault(country_key(lab), []).append(name)
        # Wave I is the one wave with no `Q1012` at all; it offers no country a religion
        # answer and so is in no country's `available`. `load` explains it where a module
        # asks for it anyway.
        if rel is None:
            continue
        df, _m = pyreadstat.read_sav(p, usecols=[c, rel])
        cname = df[c].map(clab).fillna(df[c].astype(str)).astype(str).str.strip()
        hit = cname.map(country_key).isin(want)
        if hit.any() and df.loc[hit, rel].notna().any():
            available.append(name)
    return available, declared


def fold(label):
    """The differences between two wordings that carry no meaning: case, spacing, edge marks.

    Deliberately narrow, and the narrowness is the point. `Other` and `other` fold together
    because nothing but the shift key separates them. `Other` and `Something else:
    SPECIFY_______` do NOT, and neither do `Refused to answer` and `refused`, because deciding
    that two different sentences are one box on one showcard is a reading of the questionnaires
    and belongs to whoever builds that country. This function exists to DETECT a collision, not
    to resolve one; see `assert_one_wording`.

    **The one addition, 2026-09-08: a leading `<n>.` is stripped.** Wave II spells its `Q1012`
    answers `1. muslim`, `2. christian`, `99999. declined to answer` — the numeric code
    repeated inside the label, exactly as it does for the country (see `country_key`). Without
    this, `1. muslim` and `Muslim` are two categories that DO NOT COLLIDE, so
    `assert_one_wording` says nothing and the pool silently carries Islam twice with every
    total still adding up, which is the failure that function exists to catch. Stripping the
    ordinal is mechanical and cannot merge two different answers: it is the same character
    class the country column needed, and it leaves `Something else: SPECIFY_______` and `Other`
    as far apart as they were.
    """
    s = unicodedata.normalize("NFKC", str(label))
    s = _ORDINAL_PREFIX.sub("", " ".join(s.split()))
    return s.strip(" .,:;!?_-").casefold()


def assert_one_wording(df, country, cat_col="category"):
    """ONE ANSWER, ONE SPELLING, or this refuses to pool and hands the call back.

    The twin of the module docstring's code-re-use guard, one column across, and the failure
    it catches is the one the codes' guard cannot see. `load()` decodes each wave through that
    wave's own labels, so no numeric code survives into the frame — but from then on the
    pooling key is the label STRING, and nothing about `pd.concat` notices that wave V spells
    an answer differently from wave IV.

    Lebanon is the measured case and it is one shift key wide: `Other` in waves IV, VI-2 and
    VIII (194 respondents) and `other` in wave V (190). One box on one card, arriving as two
    categories. What that costs, in order of how quietly it happens:

      * `national()` reports two answers where the questionnaire has one;
      * each is ranked by `stability()` on half the respondents, so the test that decides
        whether a category carries its own geography runs at half power;
      * each is measured against `ELIGIBLE_FLOOR` separately, so an answer that clears 1%
        whole can be dropped as too small to place, twice;
      * `taxonomy/<cc>*.py`'s `MAP` needs both spellings or one silently resolves to `None`.

    **Every one of those preserves the totals**, which is why nothing downstream catches it.

    THIS DOES NOT FOLD THE DATA, and that is not an oversight. Merging the two spellings is
    almost certainly right for Lebanon and is still a judgement: it asserts that two waves used
    one wording for one answer, which is a reading of two showcards rather than a typo, and the
    module cannot tell that case from two waves using similar wordings for two different
    answers. So it raises, names both spellings and the waves each came from, and leaves the
    merge to the country module, where it can be written down next to the reason.

    Note what is NOT asserted: that every wave offers the same answers. It does not, the module
    docstring says why, and Egypt would fail such a check on wave V's `Atheist` alone. A
    missing answer is a different card; two spellings of one answer are the same card twice.
    """
    seen = {}
    for (lab, wave), n in df.groupby([cat_col, "wave"], sort=False).size().items():
        seen.setdefault(fold(lab), {}).setdefault(lab, []).append((wave, int(n)))
    clashes = {f: v for f, v in seen.items() if len(v) > 1}
    if not clashes:
        return
    lines = []
    for f, spellings in sorted(clashes.items()):
        lines.append(f"    {f!r} arrives as {len(spellings)} categories:")
        for lab, waves in sorted(spellings.items()):
            waves.sort(key=lambda wn: WAVE_ORDER.get(wn[0], 99))
            where = ", ".join(f"wave {w} n={n}" for w, n in waves)
            lines.append(f"      {lab!r}  {sum(n for _w, n in waves)} respondents  ({where})")
    raise SystemExit(
        f"{country}'s pooled waves spell one answer more than one way, so the pool would "
        "carry it as two categories with the totals still adding up:\n"
        + "\n".join(lines)
        + "\n  Nothing is folded here on purpose: whether those really are one box on one "
        "showcard is a reading of the questionnaires, and this module cannot tell it from "
        "two similar wordings for two different answers. Decide it in the country module, "
        "re-word the `category` column there with the reason written down, and call "
        "`ab.assert_one_wording` again on the result. See the docstring for what pooling "
        "them unnoticed would cost.")


def load(country, expect_waves=None, waves=None, recode=None, omit=None, extra=None):
    """One country's respondents, decoded wave by wave through that wave's own labels.

    Returns [`wave`, `wave_no`, `category`, `geo_raw`, `geo_code`, `w`] plus one column per
    entry in `extra`, one row per
    respondent who answered `Q1012`. `category` is the ANSWER'S OWN WORDING (spec §2.4) and
    never a numeric code; see the module docstring for why that is not a style preference.

    `expect_waves` is the list of wave names the country is known to appear in, asserted so a
    re-release that adds or drops a wave is a failure here rather than a quiet re-levelling.
    **It is checked against `wave_coverage`, which reads the files, and not against the pooled
    frame.** Until 2026-09-09 it was checked against the frame — after that frame had been
    filtered by `waves=` — so a wave the pool never asked for could not appear on either side
    of the comparison and the "adds" half of that sentence could not fail. `wave_coverage`'s
    docstring is the write-up.

    ## `waves` — WHICH WAVES THE COUNTRY MODULE CHOSE, AND THE DIFFERENCE FROM `expect_waves`

    `expect_waves` asserts what the FILES contain. `waves` decides what the POOL contains, and
    a wave left out of it is a decision the country module has to justify. Two reasons a
    country needs it, both real:

      * **the wave asked something else.** Wave I's religion item is `q711` and not `q1012`,
        and it is the only wave whose questionnaire is numbered the old way; more to the point
        wave I carries **no subnational variable at all** — 181 columns, `country` and nothing
        finer — so it cannot enter a pool that is cut by governorate however it is decoded.
        Left out with `waves=`, it is a stated choice; left to the `rel is None` check below,
        it is a hard failure that reads as a broken file.
      * **the country is already drawn from a pool that did not have it.** Egypt was built
        before `country_key` made wave II visible, so `sources/eg.py` pins its four waves and
        Egypt's numbers do not move underneath a published map.

    ## `omit` — AND WHY A WAVE THE FILES OFFER CANNOT JUST BE LEFT OUT

    Only the first of those two reasons is visible in `waves=` on its own. Wave I is absent
    from every country's `available` because it has no `Q1012`, so leaving it out states
    nothing that the files do not already say. The second is different: wave II is in the
    files, with a religion answer, and Egypt's pool does not have it. That is the wave II
    failure's own shape, and a pool is not allowed to be quietly narrower than the files.

    So every wave in `wave_coverage`'s `available` must be either in `waves=` or named in
    `omit`, which is `{wave: "the reason"}` and is asserted the way `recode` is: a wave named
    here that the files do not offer is a failure, not a stale line to ignore. `OMITTED`
    carries Egypt's, for the reason written beside it.

    ## `recode` — TWO SPELLINGS OF ONE ANSWER, MERGED WHERE THE REASON CAN BE WRITTEN DOWN

    `{raw label: replacement}`, applied to `category` immediately before `assert_one_wording`,
    which is the point that function's own error message asks the country module to act at.
    Every key must occur in the pool or this raises, so a re-release that fixes a spelling
    turns a stale entry into a failure rather than into silence.

    It is deliberately NOT a widening of `fold`. Jordan needs `refused` (wave V) and
    `Refused to answer` (waves VI-2 and VII) to be one answer; a fold loose enough to merge
    that pair on its own would also merge Lebanon's `Something else: SPECIFY_______` into its
    `Other`, which is two answers and not one. A dictionary in the country module, beside the
    sentence saying why, is the difference.

    ## `extra` — A SECOND ANSWER COLUMN, DECODED THE SAME WAY, ADDED 2026-09-09 FOR IRAQ

    `{column name: (alias, alias, ...)}`. For each wave the first alias present is decoded
    through **that wave's own labels** and lands in the frame under `column name`; a wave with
    none of the aliases contributes NaN and is named in the printed line, because "this wave
    did not ask" and "this wave asked and nobody answered" are different facts and only the
    first is visible here.

    It exists because `Q1012` is not the finest religion answer everywhere. Iraq's is the sect
    follow-up: `Q1012A` in waves V and VI-3, renamed `Q1012A_MUSLIM` in VII and VIII, and it
    is what puts Sunni and Shia on the map. Reading it in the country module instead would
    mean re-implementing the per-wave decode this function exists to centralise, and the
    module docstring's first section is about what happens when that decode is skipped.

    **It decodes and nothing else.** Which answers on the second column belong together, which
    leave the universe, and how the two columns compose into one `category` are all the
    country module's, because they are readings of two questionnaires rather than of a file.
    `assert_one_wording` is NOT re-run afterwards for the same reason: the composed category is
    the country module's construction, so that module calls it again on the result, which is
    what this one's error message asks for anyway.
    """
    import pyreadstat

    want = None if waves is None else list(dict.fromkeys(waves))
    known = list(WAVE_NAMES)
    if want is not None:
        unknown = [w for w in want if w not in known]
        if unknown:
            raise SystemExit(f"waves= names {unknown}, which are not Arab Barometer waves; "
                             f"the file list is {known}")
    omitted = dict(OMITTED.get(country_key(country), {}))
    omitted.update(omit or {})
    unknown = [w for w in omitted if w not in known]
    if unknown:
        raise SystemExit(f"omit names {unknown}, which are not Arab Barometer waves; "
                         f"the file list is {known}")

    # Read the files before pooling anything: which waves offer this country a religion answer,
    # and whether the file also spells the country a second way. Both are questions the pooled
    # frame cannot answer about itself, which is why the old `expect_waves` could not fail.
    available, declared = wave_coverage(country)
    assert_no_near_miss(country, declared)
    spellings = sorted(country_spellings(country))
    print(f"  the files offer {country} a religion answer in waves "
          f"{', '.join(available) or 'NONE'}"
          + (f"  (matched on {spellings})" if len(spellings) > 1 else ""))

    country_keys = country_spellings(country)
    frames = []
    for name, ordinal, _zipname, savname in WAVES:
        if want is not None and name not in want:
            continue
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
        sub = df[cname.map(country_key).isin(country_keys)]
        if not len(sub):
            continue
        if rel is None:
            raise SystemExit(
                f"wave {name} has {country} rows but no Q1012. Wave I asks religion as "
                "`q711` and carries no subnational variable at all, so it cannot be pooled "
                "with the others; if that is the wave here, leave it out with `waves=` and "
                "say why, rather than reaching for q711.")
        rlab = meta.variable_value_labels.get(rel, {})
        glab = meta.variable_value_labels.get(geo, {}) if geo else {}
        undecoded = sorted(set(sub[rel].dropna()) - set(rlab))
        if undecoded:
            raise SystemExit(f"wave {name}: Q1012 codes with no label: {undecoded}")
        frame = pd.DataFrame({
            "wave": name,
            "wave_no": ordinal,
            "category": sub[rel].map(rlab),
            "geo_code": sub[geo] if geo is not None else np.nan,
            "geo_raw": (sub[geo].map(glab) if geo is not None else np.nan),
            "w": pd.to_numeric(sub[wt], errors="coerce") if wt else 1.0,
        })
        note = ""
        for col, aliases in (extra or {}).items():
            src = _col(df, *aliases)
            if src is None:
                frame[col] = np.nan
                note += f"  [{col}: not asked in this wave]"
                continue
            xlab = meta.variable_value_labels.get(src, {})
            bad = sorted(set(sub[src].dropna()) - set(xlab))
            if bad:
                raise SystemExit(f"wave {name}: {src} codes with no label: {bad}")
            frame[col] = sub[src].map(xlab)
            note += (f"  [{col}={src}, {int(sub[src].notna().sum())} answered]")
        frames.append(frame)
        print(f"  wave {name:<5} n={len(sub):>6}  Q1012 answered by "
              f"{int(sub[rel].notna().sum()):>6}  "
              f"{'' if geo is None else str(sub[geo].map(glab).nunique()) + ' Q1 labels'}"
              + note)
    if not frames:
        raise SystemExit(f"no {country} rows in any wave")
    out = pd.concat(frames, ignore_index=True)
    out = out[out["category"].notna()].copy()
    out["w"] = out["w"].fillna(1.0)
    if out.empty:
        raise SystemExit(
            f"{country} has rows in these files and not one of them answered Q1012, so there "
            "is nothing to pool. Saudi Arabia is the measured case and it is a fact about the "
            "survey rather than a fault here: 1,404 wave II respondents with a governorate "
            "and a weight and an entirely empty religion column, and no rows at all under "
            "wave V's `Kingdom of Saudi Arabia`. Mauritania's 3,200 respondents in waves VII "
            "and VIII are the same. Those countries cannot be drawn from this instrument.")

    got = [w for w in WAVE_NAMES if w in set(out["wave"])]
    # Two code paths reading the same files: `available` from `wave_coverage`'s two-column
    # scan, `got` from the full read. They must agree on the selection actually asked for.
    should = [w for w in available if want is None or w in want]
    if got != should:
        raise SystemExit(f"the pool holds waves {got} for {country} and the file scan says "
                         f"{should} — the two reads of the same files disagree, so one of "
                         "them is wrong and the pool cannot be trusted. STOP.")

    hidden = [w for w in available if w not in should and w not in omitted]
    if hidden:
        raise SystemExit(
            f"the files offer {country} a religion answer in waves {hidden} and `waves=` "
            f"leaves them out of the pool, saying nothing. That is the wave II failure's own "
            "shape (see `country_key`): a wave in the file, absent from the pool, and no "
            "warning. If leaving them out is right, name each one in `omit={wave: reason}` "
            "and the reason is then on the page beside the pool; if it is not right, add "
            "them to `waves=`.")
    stale = [w for w in omitted if w not in available]
    if stale:
        raise SystemExit(
            f"omit names waves {stale}, which the files do not offer {country} at all. "
            "Either a re-release has changed the pool or the entry was never right; do not "
            "delete it without reading what it was for.")
    if omitted:
        for w in WAVE_NAMES:
            if w in omitted:
                print(f"  wave {w} is in the files and deliberately out of the pool: "
                      f"{omitted[w]}")

    if expect_waves is not None:
        claim = [w for w in WAVE_NAMES if w in set(expect_waves) | set(omitted)]
        if available != claim:
            raise SystemExit(
                f"the files offer {country} a religion answer in waves {available}; "
                f"`expect_waves` plus the declared omissions account for {claim}. A "
                "re-release has added or dropped a wave, which is a re-levelling of the pool "
                "and not a detail — read what changed before rebuilding anything.")

    for raw, replacement in (recode or {}).items():
        n = int((out["category"] == raw).sum())
        if not n:
            raise SystemExit(
                f"recode names {raw!r}, which {country}'s pool does not contain. Either a "
                "re-release has changed the wording or the entry was never right; do not "
                "delete it without reading what it was for.")
        print(f"  recoded {n} respondent(s) from {raw!r} to {replacement!r}")
        out.loc[out["category"] == raw, "category"] = replacement

    assert_one_wording(out, country)
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

    ## A SMALL COUNTRY IS CHECKED EXHAUSTIVELY, NOT SAMPLED

    `sources/lapop.py`'s version, which this was copied from, samples `n_perm` orderings and
    fails if ANY of them reaches the observed r. That rule is right where the orderings vastly
    outnumber the draws — its own countries have 14, 22 and 23 units, so 14! = 8.7e10 against
    20,000 draws — and it breaks quietly where they do not. Lebanon's Arab Barometer cut is
    about eight governorates, 8! = 40,320, so 20,000 draws return the CORRECT ordering about
    every other run and a perfect decode hard-fails on nothing but unit count.

    Two things follow, and neither of them is a loosened bar:

      * **the observed ordering is not part of the null.** The null is the wrong joins; the
        right one is what is being tested. It is excluded by VALUE rather than by index, which
        also excludes the orderings that only swap units of equal population — those reproduce
        the observed r exactly and no correlation can tell them from the truth, so counting
        them as beating it is a false alarm rather than a catch.
      * **below `EXACT_PERM_MAX` orderings, every one of them is checked.** At eight units the
        result is then a proof and not a sample: *none of the 40,319 other ways of pairing
        these units reaches this r.* That is a stronger statement than 20,000 draws can make,
        and it costs about a second.

    The ceiling on that statement is printed, because it is the thing a reader should judge:
    `n` units can express at best 1 in n!-1, so seven units is 1 in 5,039 and five is 1 in
    119. **Below about seven units this check cannot carry a join on its own** however cleanly
    it passes, and the country needs a second witness — the argument is §11ad's and it is the
    same one that stopped lapop's age check being asserted.
    """
    print("\n  held-out check (nothing here touches the religion column):")
    share_s = df.groupby(unit_col)["w"].sum() / df["w"].sum()
    share_p = pop / pop.sum()
    j = pd.concat([share_s.rename("survey"), share_p.rename("pop")], axis=1).dropna()
    if len(j) != len(share_s):
        raise SystemExit(f"{len(share_s) - len(j)} sampled units have no population")
    r = np.corrcoef(j["survey"], j["pop"])[0, 1]
    # A nan r would compare False against every permuted one and the check would pass in
    # silence. It means one of the two columns is constant, or there is one unit.
    if not np.isfinite(r):
        raise SystemExit(f"the survey/{pop_source} correlation over {country}'s {len(j)} "
                         "units is undefined, so this check cannot say anything; one of the "
                         "two shares does not vary across the units")
    ratio = (j["survey"] / j["pop"]).sort_values()
    print(f"    unit share of respondents vs {pop_source}:  r = {r:+.3f} over {len(j)} units")
    print(f"      thinnest sampled {ratio.index[0]} at {ratio.iloc[0]:.2f}x its population "
          f"share, fullest {ratio.index[-1]} at {ratio.iloc[-1]:.2f}x")

    rng = np.random.default_rng(seed)
    a, b = j["survey"].to_numpy(), j["pop"].to_numpy()
    n = len(j)
    total = math.factorial(n)
    exact = total <= EXACT_PERM_MAX
    if exact:
        pairings = [b[list(p)] for p in itertools.permutations(range(n))]
        how = f"all {total - 1:,} other orderings of the same units"
    else:
        pairings = [rng.permutation(b) for _ in range(n_perm)]
        how = f"{n_perm:,} random pairings of the same units"
    perm = np.array([np.corrcoef(a, p)[0, 1] for p in pairings])
    # By value, not by index: this drops the observed ordering and any that only swaps units of
    # equal population, which reproduce the observed r and are not wrong answers.
    same = np.array([np.array_equal(p, b) for p in pairings])
    beaten = int(((perm >= r) & ~same).sum())
    print(f"      against {how}: best r = {perm[~same].max():+.3f}, and {beaten} reach the "
          f"observed one")
    if not exact and same.sum():
        print(f"      {int(same.sum())} of those draws WERE the observed ordering and are not "
              f"part of the null ({n} units allow only {total:,} of them)")
    # Only where it constrains anything: at 24 units the ceiling is a 24-digit number and
    # saying it out loud is noise. Under a million orderings it is the thing to judge.
    if total <= 1_000_000:
        print(f"      {n} units allow {total:,} orderings, so the strongest this check can "
              f"say is 1 in {total - 1:,}"
              + ("" if n >= 7 else " — too weak to carry the join on its own, see the "
                                   "docstring"))
    if beaten:
        raise SystemExit(
            f"{beaten} of {how} match or beat the observed r={r:+.3f} for {country}. The "
            "population check does not pin this decode, so the join needs a witness that "
            "does before anything is drawn.")
    return r


def _binom_pmf(n, p):
    """The binomial pmf over 0..n, in log space so a long tail does not underflow."""
    k = np.arange(n + 1)
    if p <= 0.0:
        out = np.zeros(n + 1)
        out[0] = 1.0
        return out
    if p >= 1.0:
        out = np.zeros(n + 1)
        out[n] = 1.0
        return out
    lg = np.array([math.lgamma(i + 1) for i in range(n + 2)])
    logc = lg[n] - lg[k] - lg[n - k]
    return np.exp(logc + k * np.log(p) + (n - k) * np.log1p(-p))


def _p_equal_share(a, na, b, nb):
    """P(two INDEPENDENT samples of size na and nb return the same share), at the pooled rate.

    The exact probability, summed over every pair of counts whose shares coincide: `k/na` and
    `j/nb` are equal iff `k*nb == j*na`, so only the k that make `k*nb` divisible by `na`
    contribute. This is what makes `quota_agreement` a test rather than a threshold: the
    chance of an exact tie is a computable number and it depends on how big the two samples
    are, which no fixed "how many cells agree" rule can know.
    """
    p = (a + b) / (na + nb)
    pa, pb = _binom_pmf(na, p), _binom_pmf(nb, p)
    tot = 0.0
    for k in range(na + 1):
        num = k * nb
        if num % na == 0:
            j = num // na
            if 0 <= j <= nb:
                tot += pa[k] * pb[j]
    return float(min(max(tot, 1e-12), 1.0))


def _poisson_binomial_tail(ps, k):
    """P(at least k of these independent Bernoullis succeed), by convolution."""
    dist = np.array([1.0])
    for p in ps:
        dist = np.convolve(dist, [1.0 - p, p])
    return float(dist[k:].sum())


def quota_agreement(df, unit_col="geo_id", cat_col="category", verbose=True):
    """DID THE FIELDWORK SET THE ANSWER? Every pair of waves, compared cell by cell.

    ## THE FAILURE THIS EXISTS FOR, AND WHY §14.16 CANNOT SEE IT

    **Lebanon, 2026-09-09.** Arab Barometer's Lebanese sample is a fixed sect-by-governorate
    quota: the contractor is given a number of Sunni, Shia, Maronite, Orthodox, Catholic and
    Druze interviews to fill in each governorate, and fills it. Wave V (2018-19) and wave VII
    (2021-22) return **the same Christian count in all eight governorates** — Akkar 30 of 160,
    Beirut 90 of 250, North 100 of 330, South 10 of 260, Mount Lebanon 650 of 960, Baalbek 0 of
    150, Nabatieh 0 of 140 — three years and two fieldwork rounds apart. Waves VI-1, VI-2 and
    VI-3 share a second grid: Kesrwan-Jbeil comes back **100% Christian three times**, Akkar
    exactly 10 Christians in 70 three times. `sources/lb.md` has the tables.

    **The split-half test passes such a country perfectly and that is exactly the problem.**
    §14.16 asks whether a category's ranking across units REPLICATES between the early and the
    late waves. A quota replicates by construction — it is the same grid applied twice — so the
    one test this project uses to decide whether a category carries its own geography returns
    its strongest possible verdict on the one input where the geography is not a measurement at
    all. A country drawn that way would be a map of the pollster's assumption about where the
    sects live, at the resolution the pollster assumed it at, and nothing downstream would
    disagree with it. THIS IS THE CHECK THAT HAS TO RUN FIRST.

    ## THE STATISTIC

    For each pair of waves, take the units both sampled and the answers both cards offered,
    keeping a unit only where the shared answers cover at least 95% of that unit's respondents
    in both waves — otherwise a wave whose card had no `Druze` box is being compared with one
    that had. Within a unit, drop the largest answer: the shares sum to one, so the last cell
    is not a free comparison. What is left is a set of cells where two independent samples each
    measured a share, and `_p_equal_share` gives the exact probability that they would land on
    the same rational number. The number of exact agreements is then read against the
    Poisson-binomial tail of those probabilities.

    Returns `(p, wave_a, wave_b, hits, cells)` for the most extreme pair, or `None` if no pair
    was comparable. The p-value is per pair; `assert_not_quota` applies the Bonferroni.

    ## RUN IT ON EVERY RELIGION COLUMN THE FILE OFFERS, NOT ONLY THE ONE YOU MEAN TO DRAW

    Lebanon's religion column comes in at an adjusted **1.4e-4** and its sect column at
    **1.1e-3**, which is the wrong side of the bar by a hair — not because the sect column is
    cleaner, but because the waves that carry it do not include wave VII, and wave VII is half
    of the most damning pair. **The quota is a property of the FIELDWORK and not of a column.**
    A country that fails this on any of its religion columns has failed it, and a pass on the
    narrower pool is not evidence about the wider one.
    """
    tabs = {w: pd.crosstab(d[unit_col], d[cat_col]) for w, d in df.groupby("wave")}
    order = [w for w in WAVE_NAMES if w in tabs]
    worst, n_pairs = None, 0
    if verbose:
        print("\n  quota check: does any pair of waves return the SAME composition per unit?")
    for i in range(len(order)):
        for jx in range(i + 1, len(order)):
            wa, wb = order[i], order[jx]
            A, B = tabs[wa], tabs[wb]
            units = sorted(set(A.index) & set(B.index))
            cats = sorted(set(A.columns) & set(B.columns))
            if len(units) < 3 or len(cats) < 2:
                continue
            AA, BB = A.loc[units, cats], B.loc[units, cats]
            keep = [u for u in units
                    if AA.loc[u].sum() >= 0.95 * A.loc[u].sum()
                    and BB.loc[u].sum() >= 0.95 * B.loc[u].sum()]
            if len(keep) < 3:
                continue
            AA, BB = AA.loc[keep], BB.loc[keep]
            use = [c for c in cats if c != AA.sum().idxmax()]
            ps, hits = [], 0
            for u in keep:
                na, nb = int(AA.loc[u].sum()), int(BB.loc[u].sum())
                if na < 5 or nb < 5:
                    continue
                for c in use:
                    a, b = int(AA.loc[u, c]), int(BB.loc[u, c])
                    if (a == 0 and b == 0) or (a == na and b == nb):
                        continue          # degenerate: agreeing on nobody says nothing
                    ps.append(_p_equal_share(a, na, b, nb))
                    hits += int(a * nb == b * na)
            if len(ps) < 4:
                continue
            n_pairs += 1
            p = _poisson_binomial_tail(ps, hits)
            if verbose and (p < 0.05 or worst is None or p < worst[0]):
                print(f"    {wa:>5} vs {wb:<5} {len(keep):>2} units, {len(ps):>3} free cells, "
                      f"{hits:>3} identical to the interview   p={p:.3g}")
            if worst is None or p < worst[0]:
                worst = (p, wa, wb, hits, len(ps))
    if worst is None:
        if verbose:
            print("    no two waves are comparable, so this says nothing")
        return None, 0
    if verbose:
        print(f"    most extreme pair: {worst[1]} vs {worst[2]}, {worst[3]}/{worst[4]} cells "
              f"identical, p={worst[0]:.3g} over {n_pairs} pairs")
    return worst, n_pairs


def assert_not_quota(df, country, unit_col="geo_id", cat_col="category", quota_ok=None):
    """Refuse to run the split-half on a country whose per-unit composition is fieldwork.

    `quota_ok` is a written sentence, not a flag, and it is the same escape hatch as
    `stability`'s `override`: a country that fails this and is drawn anyway must say why on the
    page. **There is no such country yet, and the one that failed is not drawn.**
    """
    worst, n_pairs = quota_agreement(df, unit_col, cat_col)
    if worst is None:
        return
    p, wa, wb, hits, cells = worst
    adjusted = min(1.0, p * max(n_pairs, 1))
    print(f"    Bonferroni over {n_pairs} pairs: p={adjusted:.3g}, bar {QUOTA_P_BAR:g}")
    if adjusted > QUOTA_P_BAR:
        return
    if quota_ok:
        print(f"    UNDER THE BAR and drawn anyway: {quota_ok}")
        return
    raise SystemExit(
        f"{country}'s waves {wa} and {wb} return the same composition in {hits} of {cells} "
        f"free cells, p={p:.3g} ({adjusted:.3g} after Bonferroni over {n_pairs} pairs). Two "
        "independent samples do not do that. This is a sect-by-unit QUOTA in the fieldwork, "
        "which means the per-unit shares are the survey firm's own assumption about where the "
        "groups live and not a measurement of it — and `stability` cannot tell you so, because "
        "a quota replicates between the wave halves perfectly. Lebanon is the measured case "
        "and is NOT drawn from this survey; `sources/lb.md` and sources.md §11al have the "
        "tables. If you believe this country is different, say so in `quota_ok=` with the "
        "reason, which puts it on the page where a reader can disagree with it.")


def stability(df, nat, n_units, unit_col="geo_id", cat_col="category", override=None,
              quota_ok=None):
    """WHICH CATEGORIES CARRY THEIR OWN GEOGRAPHY — the split-half decides it.

    Spec §14.16's test. Each category is ranked across the units in the earlier half of the
    waves and again in the later half, and the two orderings are compared. A category whose
    ranking does not replicate has not demonstrated that it HAS a geography, whatever its
    pooled spread looks like.

    **`assert_not_quota` RUNS FIRST AND THAT ORDER IS THE POINT.** Replication is evidence only
    where the two halves could have disagreed. If the fieldwork fills a fixed number of each
    answer in each unit, the halves are the same grid twice and this test passes at its
    strongest on a geography nobody measured. Lebanon is the measured case, 2026-09-09.

    ## THE BAR IS THE EXACT NULL — `sources/spearman_null.py` — AND IT USED TO BE AN SE

    The smallest attainable rho whose one-sided exact p-value under the permutation null is at
    most 0.05, enumerated at ten units or fewer and sampled above. **This module previously
    copied `1.96/sqrt(n-1)` out of `sources/lapop.py`, and that is a standard deviation rather
    than a 95th percentile.** On Jordan's twelve governorates it was a 0.023-level test and on
    Egypt's twenty-three units a 0.024-level one — stricter than it claimed everywhere.
    Corrected 2026-09-09 on Anita's ruling in `ask/answered/007-cr`, which was filed against
    lapop and to which Jordan's build appended the evidence that this module had inherited the
    same line. **Neither country moved**: Egypt's two answers come in at +0.495 against a bar
    that fell from +0.4179 to +0.3528, Jordan's at +0.617 against one that fell from +0.5910 to
    +0.5035, and both passed before and after. The level is 0.05 and did not change.

    It is still never moved to make something pass; correcting the arithmetic that implements a
    level is a different act from choosing a different level, and the second one is Anita's.

    **HARMONISE THE UNITS BEFORE CALLING THIS.** §11af ran it on Egypt's raw `Q1` labels and
    got +0.416 against a +0.566 bar, a clear failure — because only 13 of 27 units appeared in
    both halves and the test was being run on mangled units. Harmonised, the same data give
    +0.495 against +0.418 and pass. A stability test run before the units exist reports noise
    as a negative, and a negative is what this project treats as evidence.

    `override` is `{category: "the reason"}` and is a person's decision, not an agent's; see
    `sources/lapop.py`'s docstring for when it is even available.
    """
    override = override or {}
    # BEFORE ANY CORRELATION. A quota replicates between the halves perfectly, so this test
    # returns its strongest verdict on the one input where the per-unit shares are not a
    # measurement at all; `assert_not_quota`'s docstring is the Lebanon case that found it.
    assert_not_quota(df, "this country", unit_col, cat_col, quota_ok=quota_ok)
    bar, how = spearman_null.critical_rho(n_units)
    waves = sorted(df["wave_no"].unique())
    cut = waves[len(waves) // 2]
    early, late = df[df["wave_no"] < cut], df[df["wave_no"] >= cut]
    print(f"\n  split-half stability across waves (§14.16), bar = +{bar:.4f} at 95% on "
          f"{n_units} units ({how}; the old 1.96/sqrt(n-1) was "
          f"+{1.96 / np.sqrt(n_units - 1):.4f}, a "
          f"{spearman_null.exact_p(1.96 / np.sqrt(n_units - 1), n_units):.3f}-level test):")
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
        # The overlap does not depend on the category — it is the units that have respondents
        # in both halves — so a change in it is a change in the pool and the bar is now the
        # wrong bar. §11af's Egypt run is the cautionary case: it reported a bar computed on
        # one unit count and a correlation computed on another.
        if len(j) != n_units:
            raise SystemExit(f"{len(j)} units appear in both wave halves, not the {n_units} "
                             f"the bar was computed for — re-read the pool before trusting "
                             "any correlation below")
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp = j["e"].corr(j["l"], method="spearman")
            pe = j["e"].corr(j["l"])
        passed = bool(np.isfinite(sp)) and sp >= bar
        forced = c in override
        if passed or forced:
            carries.append(c)
        shown = f"{sp:+10.3f}{pe:+9.3f}" if np.isfinite(sp) else f"{'undefined':>10}{'':>9}"
        p = spearman_null.exact_p(sp, n_units)
        if passed:
            verdict = f"own geography  ({len(j)} units in both halves, p={p:.3f})"
        elif forced:
            verdict = "own geography — UNDER THE BAR, drawn on Anita's call"
        else:
            verdict = ("NOT distinguishable from zero"
                       + (f" (p={p:.3f})" if np.isfinite(sp) else ""))
        print(f"    {str(c)[:26]:<28}{nat[c] * 100:9.2f}%{shown}  {verdict}")
        # The exact null has no ties; average-ranked shares need the conditional one. Egypt
        # ties five units and Jordan two, both were run against the conditional null on
        # 2026-09-09, and neither verdict changed. See `spearman_null`'s ## TIES.
        tied = spearman_null.ties_note(j["e"], j["l"])
        if tied:
            print(f"       {tied}")
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
