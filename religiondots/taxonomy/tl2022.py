"""INETL 2022 census religion -> religiondots taxonomy.

Nine categories at municipality level, out of the thirteen `<Municipality> em Números 2022`
volumes plus Atauro as the residual against main report table 4.07 (`sources/tl.py`). The
universe is the population aged 3 and over in private households, 1,248,705 people.

    97.474%  Catolica                     -> christianity.catholic
     2.043%  Protestante/Evangélica       -> christianity.protestant
     0.256%  Musulmano                    -> islam
     0.082%  Seluk                        -> other.tl
     0.064%  Laiha relijiaun              -> unaffiliated
     0.030%  Budista                      -> buddhism
     0.019%  Tradicional                  -> indigenous
     0.019%  La hatan                     -> EXCLUDED, the non-response cell
     0.013%  Hindu                        -> hinduism

**THIS IS THE MOST LOPSIDED CATEGORY LIST ON THE MAP AND THE INTERESTING PART IS ONE
ISLAND.** Catholicism is 97.5% of the country and above 96% in twelve of the fourteen
municipalities. The exception is Atauro, 9,622 people on an island 25 km off Dili, which is
**55.4% Protestant and 44.5% Catholic** and is the only unit anywhere in Timor-Leste where
Catholicism is not the answer of nine people in ten. Aileu is second at 7.9% Protestant and
nothing else reaches 4%.

**NO CHRISTIAN BODY IS NAMED ANYWHERE.** The census offers `Catolica` and
`Protestante/Evangélica` and stops, so this is a Germany-shaped country in the sense
`sources.md` §11b means: a very large Catholic cell with no diocese, order or rite behind it,
and a Protestant cell that pools the historic Reformed church, the Assemblies of God and
every later arrival into one number.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "No answer":
        "the census's own non-response cell, `La hatan` in the Baucau volume and `No "
        "answer` in the other twelve. 239 people, 0.019% of the universe, the smallest "
        "non-response cell on this map by a wide margin. Not drawn, per §3.5: it is a "
        "property of the enumeration and not an answer anybody gave. It is in `gap=` "
        "beside the under-threes.",
}

REVIEW = {
    "Catholicism":
        "-> christianity.catholic, with no rite and no body under it. **1,217,157 people, "
        "97.474%, the highest Catholic share of any country on this map**, ahead of "
        "Paraguay at 90.5% and Poland at 90.1%. The census names nothing below the word: "
        "no diocese, no order, no "
        "distinction between the Latin rite everyone in Timor-Leste actually belongs to and "
        "anything else, so the cell sits on the undivided Catholic node the way "
        "`bd2011.py`'s Islam sits on its root. "
        "Its range across the fourteen municipalities is 99.8% in Covalima down to 44.5% on "
        "Atauro, and if Atauro is set aside the whole rest of the country fits between "
        "91.9% (Aileu) and 99.8%.",
    "Protestantism/Evangelicalism":
        "-> christianity.protestant, the `named no body` node, NOT christianity and not "
        "`christianity.evangelical`. 25,511 people, 2.043%. "
        "**The census prints one cell for both words.** The 2022 questionnaire's answer is "
        "`Protestante/Evangélica` and the main report's table 4.07 heads the column "
        "`Protestantism/ Evangelicalism`, so the historic Reformed church that has been on "
        "Atauro since the Dutch-era mission and the Pentecostal and Assemblies-of-God "
        "congregations that arrived after 1999 are one number and cannot be separated. "
        "`christianity.evangelical` would assert the second reading over the first, which "
        "is the call `sv2023.py` and `gt2023.py` had the evidence to make and this census "
        "does not support, because it never asks the follow-up. So the whole cell goes to "
        "the unspecified node, which is `id2010.py`'s call for Indonesia's `Kristen` next "
        "door. "
        "**Its geography is one island and one district.** Atauro is 55.4% and Aileu 7.9%; "
        "the other twelve municipalities run from 3.4% down to 0.13%.",
    "Islam":
        "-> islam, the root, with no school or sect. 3,202 people, 0.256%. Concentrated in "
        "Dili (1,964, 0.647% of the municipality) and, more unusually, in Lautém at the "
        "eastern end of the island (363, 0.559%). Timor-Leste's Muslims are a mix of the "
        "long-settled Arab-descended community around the Mesquita An-Nur in Dili and "
        "people of Indonesian origin; the census names neither and none is inferred.",
    "Buddhism":
        "-> buddhism, with no school. 378 people, 0.030%, and 288 of them are in Dili. This "
        "is the Chinese-Timorese community, which was much larger before 1975.",
    "Hinduism":
        "-> hinduism. 161 people, 0.013% — the smallest named religion in the country. "
        "124 in Dili. "
        "**Its one oddity is worth recording**: the 2015 census put 44 of Timor-Leste's 272 "
        "Hindus in Oecusse, the exclave inside Indonesian West Timor, where the 2022 census "
        "finds one. Nothing published explains the difference and it is 43 people, so it is "
        "noted rather than acted on.",
    "Indigenous religion":
        "-> indigenous, the undivided root, and not a new node. 240 people, 0.019%. "
        "**Read this as a floor, and the census's own report says so more strongly than "
        "usual.** Section 3.2.2 of the main report observes that indigenous religion is "
        "over-represented above age 55 and that the 2015 count of just over 900 fell to "
        "under 300 in 2022, and concludes in its own words that *this type of religion is "
        "close to disappearing*. What the box counts is people who gave *lulik* instead of "
        "a church rather than the very much larger number who keep both, which is the "
        "standing caution `bw2011.py` states for Badimo and `sources.md` §11b for the "
        "continent. Timor-Leste's ancestral practice is not a competitor to Catholicism "
        "there and is not measured by a question that makes it one. "
        "No `indigenous.timor` leaf is minted for 240 people (§2.4): the count is a rounding "
        "error on the map and a single-country node would add a legend row for it.",
    "No religion":
        "-> unaffiliated. 797 people, 0.064%. **Only the Philippines at 0.040%, Kiribati at "
        "0.046% and Myanmar at 0.060% are lower**, among the 84 countries here that record "
        "any such answer at all; the rest have no such category and read as zero for a "
        "different reason entirely. It is a new box in 2022; neither the 2010 nor the 2015 "
        "census "
        "offered one, which is why the volumes print a dash in those columns. "
        "Its geography is unremarkable at this size, running from 0.164% in Liquiçá to "
        "0.001% in Oecusse.",
    "Other":
        "-> other.tl. 1,020 people, 0.082%. Genuinely a tail rather than a store cupboard: "
        "Islam, Buddhism, Hinduism and indigenous religion all have their own boxes above "
        "it and no religion has one below it, so what is left is the Baha'is, the Jehovah's "
        "Witnesses, the Latter-day Saints and anything the enumerator could not place. "
        "Per §3.11, never merged with another country's residual.",
}

MAP = {
    "Catholicism": "christianity.catholic",
    "Protestantism/Evangelicalism": "christianity.protestant",
    "Islam": "islam",
    "Buddhism": "buddhism",
    "Hinduism": "hinduism",
    "Indigenous religion": "indigenous",
    "No religion": "unaffiliated",
    "Other": "other.tl",
}


# NO `COLUMNS` DICT, AND THIS IS THE OTHER REASON A COUNTRY SHOWS UP IN check_rollup.py.
# Atauro's 9,620 drawn people are `derived` and have no measured ancestor, which is normally the
# sign that a mapping needs a COLUMNS entry naming the coarser column its rows were split out of
# (spec §7a-i-1). There is no such column here and one must not be invented: Atauro is derived
# because the CENSUS DID NOT PUBLISH THE UNIT, not because a category was split. Its `Catholicism`
# row is the source's own `Catholicism`, arrived at by subtracting thirteen municipalities from
# the national table, so the roll-up target for it would be itself. Adding COLUMNS here would
# claim a coarser measurement at that unit that does not exist. 0.77% of the country, well under
# check_rollup.py's 2% line, and the readout already calls Atauro derived.


def _key(cat):
    # Whitespace only, NOT case. `tools/check_mapping.py` borrows this function to fold
    # EXCLUDED, and a lowercasing _key makes it report a deliberately excluded universe row
    # as an unmapped category. The nine labels are minted by sources/tl.py, so exact case is
    # something this file can rely on.
    return " ".join(str(cat).split())


_FOLDED = {_key(k): v for k, v in MAP.items()}
_FOLDED_EXCLUDED = {_key(k) for k in EXCLUDED}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = str(cat)
    if c in EXCLUDED or _key(c) in _FOLDED_EXCLUDED:
        return None
    if c in MAP:
        return MAP[c]
    return _FOLDED.get(_key(c))
