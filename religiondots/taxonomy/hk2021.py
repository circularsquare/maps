"""Hong Kong — what each 2021 census ethnicity may imply, and what the survey's answers mean.

Hong Kong is mainland China's problem a second time: **no census here has ever asked about
religion**, so nothing in the count is a religious answer and everything drawn is either
spec §14.5's derivation from an ethnic category or §14.16's survey carved out of the grey.
`sources/hk.py` builds both inputs and `countries.py::_hk_counts` combines them.

TWO LAYERS, THE SAME TWO CHINA HAS.

  1. **`shares()` below** — the ethnic derivation, at the 18 District Council districts. Three
     nationalities carry a fractional share of one religion each and the remainder of every
     one of them goes to `unknown`. Every row here is `modelled`, because the coefficient is
     another country's census rather than this one's.
  2. **`SURVEY` in `sources/hk.py`** — self-identification, for the whole territory, from the
     Hong Kong Political Culture Survey 2021. `SURVEY_NODES` below says where each of its
     answers lands.

WHY THE COEFFICIENTS ARE NOT LOOKED UP ANYWHERE. **All three come from this map's own drawn
countries** — Indonesia, Pakistan and the Philippines are each built here from that country's
own census, so Hong Kong's Indonesians are given the Muslim share of Indonesia as this project
already draws it. That is not a new source, it cannot drift away from the rest of the map, and
it inherits every later correction to the source country:

    id  islam                        87.51%
    pk  islam                        96.47%
    ph  christianity.catholic.latin  78.88%

Recompute them by calling `countries.COUNTRIES[cc]["counts"]()` and grouping on node.

**AND THAT ONLY FIXES THE PROVENANCE, NOT THE APPLICABILITY — READ THIS BEFORE COPYING THE
TECHNIQUE.** Anita, 2026-09-08: *"i think doing diaspora coefficients from maps own drawn
countries is probably pretty bad for large origin countries cuz the people migrating are
probably skewed in some way."* Right, and it is §14.12: migration selects on region, class and
ethnicity, which are the axes religion varies on. A national share is the correct number for
the origin country and the wrong number for the stream that left it, and the bigger and more
religiously varied the origin, the worse it gets.

spec §14.24 makes it a permission with conditions rather than a default. **At least one must
hold, and the row should say which:**

  1. the origin share is NEAR 1, so no plausible selection moves it much — Pakistani, 96.47%;
  2. the DIRECTION of the selection is known and stated — Indonesian and Filipino, both of
     which are floors rather than estimates (see REVIEW);
  3. there is an INDEPENDENT CHECK on the result — here the survey, agreeing within 20% on the
     Muslim total.

When none holds, refuse the derivation instead of sourcing the number better. That is exactly
what happens to `Indian` and `Nepalese` in NOT_ASSERTED below, and no amount of good provenance
for India's or Nepal's national share would have rescued either.
"""

# ---------------------------------------------------------------------------------
# the ethnic derivation
# ---------------------------------------------------------------------------------
#
# Each entry is (node, share). The remainder of the nationality goes to `unknown`, and BOTH
# halves are `modelled` — §14.9's fractional shape, exactly as China's six Yunnan border
# peoples are handled.
DERIVED = {
    "Indonesian": ("islam", 0.8751),
    "Pakistani": ("islam", 0.9647),
    "Filipino": ("christianity.catholic.latin", 0.7888),
}

UNKNOWN_NODE = "unknown"

# `MAP` exists for taxonomy/registry.py's contract, which every mapping module has to meet,
# and it is NOT the whole answer here any more than `cn2000.MAP` is China's. **`shares()` is
# the resolver.** This dict is the one-node view of DERIVED above; it knows nothing about the
# `unknown` remainder each nationality carries, and nothing at all about the survey layer,
# which lives in `SURVEY_NODES` and is applied downstream in countries.py::_hk_counts.
# `coverage.py` therefore has an explicit Hong Kong branch, for both of those reasons.
MAP = {cat: node for cat, (node, _share) in DERIVED.items()}

# What each answer in the survey is. `Islam` is deliberately absent — see REVIEW.
SURVEY_NODES = {
    "Buddhism": "buddhism.mahayana",
    "Taoism": "daoism",
    "Hinduism": "hinduism",
    "Sikhism": "sikhism",
    "Protestant": "christianity.protestant",
    "Catholic": "christianity.catholic.latin",
}

# Survey answers that are NOT drawn, with the reason.
SURVEY_NOT_DRAWN = {
    "Islam":
        "2.38%, 89 respondents. **Taken from the ethnic derivation instead, which is China's "
        "rule for the same category and for the same reason.** The census counts 142,065 "
        "Indonesians and 24,385 Pakistanis exactly and says which district each lives in; the "
        "survey has 89 Muslims in a 72-cluster sample and no usable geography. The two agree "
        "on the size to within 16% (147,845 against 176,431), which is what makes either one "
        "believable — but only one of them can say that Yau Tsim Mong and Yuen Long are where "
        "the Muslims are. Its share is left inside the residual rather than reallocated, "
        "which understates the other six by about 0.8 percentage points between them.",
    "NoReligion":
        "65.83%, 2,462 respondents, and it stays `unknown` rather than becoming an "
        "irreligion node. **The same table says why**: 2,097 of those 2,462 — 56.07% of the "
        "whole sample — report practising folk religion anyway. So five sixths of Hong Kong's "
        "'no religion' is people who tend graves, burn incense and visit temples and will not "
        "call it a religion, which is the mainland's §14.22 gap measured here by one "
        "instrument instead of borrowed from Pew. Drawing it as irreligion would be a "
        "claim the source refuses to make.",
}

# ---------------------------------------------------------------------------------
# the nationalities no religion is claimed for, and every absence is a decision
# ---------------------------------------------------------------------------------
NOT_ASSERTED = {
    "Chinese":
        "**6.65 million people, 89.7% of Hong Kong, and nothing is derived from them.** "
        "Chinese ethnicity implies no religion anywhere on this map and least of all here. "
        "What they are drawn with instead is the survey layer, which is the whole point of "
        "having one: Buddhism, Taoism, Protestantism and Catholicism reach them through "
        "self-identification and the rest stay grey.",
    "Indian":
        "42,569 people, and this is spec §14.5's religiously-mixed row rather than an "
        "oversight. India is 80% Hindu nationally, but **Hong Kong's Indian community is not "
        "India in miniature** — it is disproportionately Sindhi Hindu and Punjabi Sikh, with "
        "Muslim, Parsi and Jain minorities, in proportions nobody publishes. Applying India's "
        "national vector to it would be §14.12's error exactly: a national coefficient laid "
        "over a selected migration stream. So Hong Kong's Hindus and Sikhs come from the "
        "survey at territory grain, and this nationality claims nothing.",
    "Nepalese":
        "29,701 people, refused for the sharper version of the same reason. Nepal is 81% "
        "Hindu, but Hong Kong's Nepalese are overwhelmingly the families of Gurkha soldiers, "
        "and Gurkha recruitment drew from the Gurung, Magar, Rai and Limbu — hill groups that "
        "are far more Buddhist and Kirat than Nepal's average. **The selection is precisely "
        "along the axis the coefficient would need to be stable on.** §14.12 names this "
        "failure and this is a clean instance of it.",
    "OtherSouthAsian":
        "5,314 people, mostly Bangladeshi and Sri Lankan. Bangladesh would derive cleanly and "
        "Sri Lanka would not, and the census does not separate them. Too small to be worth a "
        "assumption that cannot be checked.",
    "White": "61,582 people. No religion follows from the category.",
    "Thai": "12,972 people. Thailand is 93% Buddhist and this is the one refusal that is a "
            "close call — see REVIEW.",
    "Japanese": "10,291 people. Japanese religious self-identification is famously low while "
                "practice is near universal, which is §3.1's basis problem in one nationality.",
    "Korean": "8,700 people. Religiously mixed at home (Protestant, Buddhist, Catholic, "
              "none), and China's Korean row is already the most arguable thing in that "
              "country. Not repeated here on an eighth of the population.",
    "OtherEthnicity":
        "the part of the census's `Others` column left after the seven named groups are "
        "taken out — Other Asian, Mixed, and everyone else. A residual by construction.",
}

EXCLUDED = {
    "Total":
        "the district's own population, not a category. Drawing it would count every district "
        "twice.",
}

REVIEW = {
    "Thai":
        "-> NOT derived, and it is the closest call in this file. Thailand is 92.5% Buddhist "
        "and `th` is drawn on this map, so the coefficient is right there. Two things stopped "
        "it. Hong Kong's 12,972 Thai residents are overwhelmingly women who came as domestic "
        "workers or through marriage, which is a selected stream (§14.12) — and unlike the "
        "Indonesian and Filipino cases there is no independent check on the result, because "
        "the survey's Buddhist cell cannot be split by ethnicity. It would add about 12,000 "
        "Mahayana-labelled dots to a Theravada population, which the taxonomy would get wrong "
        "as well. **Anita's call if it is ever wanted**; it is 12 dots.",
    "Filipino":
        "-> christianity.catholic.latin at 78.88%, and the remainder to `unknown` rather than "
        "to the Philippines' other Christians. The Philippines is another 8% Protestant, "
        "Independent and Aglipayan, and drawing only the Catholic share understates Filipino "
        "Christianity by that much. That is deliberate and is §14.9's shape: one documented "
        "share to one node, the rest unclaimed. **The thing to know before touching it**: "
        "the Philippines is also 6.4% Muslim, almost entirely in Mindanao, and Hong Kong's "
        "domestic workers are recruited from Luzon and the Visayas — so the one share that "
        "would be most wrong to apply here is the one a mechanical use of `ph` would add.",
    "Indonesian":
        "-> islam at 87.51%, and the direction of the error is known. Hong Kong's Indonesian "
        "residents are ~93% domestic workers recruited overwhelmingly from Central and East "
        "Java, which are more Muslim than Indonesia as a whole. So this coefficient is a "
        "floor and the true share is probably above 95%. It is left at the documented "
        "national figure rather than adjusted upward, because §14.12's lesson is that the "
        "adjustment which feels more careful is usually the error.",
}


# --- resolution -------------------------------------------------------------------

def _key(cat):
    return " ".join(str(cat).split())


def shares(cat):
    """Ethnicity -> [(node, share, tier)], or [] if the row is not a category.

    The list sums to 1.0 for every category except `Total`. `countries.py::_hk_counts`
    reads this; `resolve` below is the single-node view tools/check_mapping.py expects.
    """
    c = _key(cat)
    if c in EXCLUDED:
        return []
    if c in DERIVED:
        node, s = DERIVED[c]
        return [(node, s, "modelled"), (UNKNOWN_NODE, 1.0 - s, "modelled")]
    return [(UNKNOWN_NODE, 1.0, "derived")]


def resolve(cat):
    """Ethnicity -> the node it principally lands on, or None if it is not a category."""
    sh = shares(cat)
    return sh[0][0] if sh else None
