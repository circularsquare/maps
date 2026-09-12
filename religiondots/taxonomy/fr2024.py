"""ESS `rlgdnm` -> religiondots taxonomy. France's French-citizen half.

The European Social Survey asks `rlgblg` — do you consider yourself as belonging to any
particular religion or denomination — and only then `rlgdnm`, which one. So the category list
below is eight religions plus a "Not applicable" that is not a non-response at all: it is
everyone who answered no to the first question, and in France it is **the largest answer on
the form**, 51.7% of citizens.

    1   Roman Catholic                 -> christianity.catholic.latin
    2   Protestant                     -> christianity.protestant
    3   Eastern Orthodox               -> christianity.orthodox.canonical
    4   Other Christian denomination   -> christianity
    5   Jewish                         -> judaism
    6   Islam                          -> islam.sunni
    7   Eastern religions              -> other.fr
    8   Other Non-Christian religions  -> other.fr
    66  Not applicable                 -> unaffiliated
    77/88/99  Refusal / DK / No answer -> excluded, spec §3.5

THIS IS gr2024.py WITH ONE CATEGORY MOVED AND NO AUTHORED CELLS, and the second half of that
is the interesting part. Greece needed two hand-written rows — the Thracian minority ESS
cannot reach and the monastic republic no sample will ever contain. **France needs none.**
ESS's French sample is not blind anywhere the way a Greek-language sample is blind in Thrace:
it finds 11.4% Muslims in Île-de-France, 10.6% Protestants in Alsace and 2.0% Jews in
Île-de-France unaided, all of which are the right shape and roughly the right size. Where it
is weak — the banlieues — there is no published regional figure to substitute, and inventing
one is what spec §14.4's first rule forbids outright.

SEVEN ROUNDS POOL, WHICH IS TWICE GREECE'S THREE, and the category list does not collapse the
way Greece's did: every round offers and uses at least ten denominations, so no category
blinks out of existence in a single round the way `Islam` does in Greek round 11. What pooling
costs instead is time — rounds 5 to 11 span 2010 to 2024 — and the drift is smaller than that
span suggests. The "no religion" share is FLAT across the whole window (52.6, 53.8, 52.3,
49.1, 49.9, 50.8, 53.5) with no trend at all. Catholicism falls 39.3 -> 36.2 and Islam among
citizens rises 4.6 -> 6.8, the latter being naturalisation and age structure rather than
conversion. **So the pooled Muslim share of 5.25% understates the present and the pooled
Catholic share overstates it**, both by about one point, and that is in note_public.
"""

EXCLUDED = {
    "Refusal": "ESS's 77. A refusal to answer, so spec §3.5 marks it rather than filling "
               "it. Tiny in France — 0.21% of pooled citizen respondents, the smallest "
               "non-response of any survey-based country here.",
    "Don't know": "ESS's 88, and the same treatment.",
    "No answer": "ESS's 99, and the same treatment.",
}

REVIEW = {
    "Not applicable":
        "-> unaffiliated, and in France it is the LARGEST node on the map — 51.7% of "
        "citizens, ~32M people, more than Catholicism. It is not a missing value: it is "
        "everyone who answered NO to `rlgblg`. branches.py's line is whether a POSITION is "
        "stated — `unaffiliated` is a report of not belonging, `secular` is a stated "
        "non-theistic stance — and 'I do not belong to a religion' is plainly the first. "
        "**Nothing in France reaches `secular` at all**, because ESS never offers atheist "
        "or agnostic as a denomination, and that is a worse hole here than anywhere else "
        "it has been noted: France has the largest self-declared atheist population in "
        "Western Europe and this map cannot draw one dot of it. gr2024.py and ge2014.py "
        "make the same call for the same reason and with far less at stake.",
    "Roman Catholic":
        "-> christianity.catholic.latin. France is Latin rite throughout; the Eastern "
        "Catholic eparchies (Ukrainian, Maronite, Armenian, Chaldean) are real, are in "
        "Paris and Marseille, and are perhaps 100,000 people between them — ESS does not "
        "ask and they are inside this node rather than in `catholic.eastern`. Said here "
        "rather than silently rolled in.",
    "Protestant":
        "-> christianity.protestant, the parent, and the parent is doing more work in "
        "France than in most countries. It holds two populations the survey cannot "
        "separate and that are not the same object: the historic Lutheran and Reformed "
        "churches — UEPAL in Alsace-Moselle, EPUdF elsewhere, together the Protestantism "
        "of the Wars of Religion and the concordat — and the evangelical and Pentecostal "
        "sector, which is the fastest-growing religious movement in the country and is "
        "largely Caribbean, African and Roma. **The geography separates them even though "
        "the question does not**: see the Alsace note in sources/fr.md.",
    "Eastern Orthodox":
        "-> christianity.orthodox.canonical, the bare parent, and NOT the `.greek` child "
        "gr2024.py uses. Greece has one national church and France has none: its Orthodox "
        "are Russian (both patriarchates and the former Rue Daru exarchate), Romanian, "
        "Greek, Serbian, Antiochian and Georgian, no one of them dominant. Naming any of "
        "them would be an assertion. The foreign half DOES name them, because a passport "
        "carries that information and a French citizen's answer of 'Orthodox' does not.",
    "Other Christian denomination":
        "-> christianity, the root, following ru2012.py and gr2024.py. The answer names no "
        "body and rules out the three that have codes of their own. In France it is "
        "largely Jehovah's Witnesses — around 130,000, one of the country's larger "
        "non-Catholic Christian bodies and the subject of a long line of European Court of "
        "Human Rights litigation — plus the Latter-day Saints and the Adventists.",
    "Islam":
        "-> islam.sunni. France's Muslims are Sunni overwhelmingly: Maliki from the "
        "Maghreb, which is three-quarters of the origin population, and Hanafi from "
        "Turkey. **The Alevis are the exception and are not separable on this half** — "
        "perhaps 100,000-150,000 people of Turkish origin, with their own cemevi network "
        "and a live legal dispute about whether they are Muslims at all. ESS does not ask "
        "and no French source counts them, so a citizen who is Alevi is drawn Sunni here. "
        "The foreign half does better: origin_religion.py splits Turkish nationals into "
        "sunni, alevism and shia, so France's Alevi node is populated by passport-holders "
        "only and is therefore an undercount by construction. gr2024.py records the same "
        "shape for the Bektashi Pomaks.",
    "Eastern religions":
        "-> other.fr with 'Other Non-Christian religions', for hr2021.py's, ru2012.py's and "
        "gr2024.py's reason: the tree has no node for 'some Eastern religion, unspecified' "
        "and picking one would invent a fact. Buddhism and Hinduism are not offered "
        "separately by ESS. **In France this costs more than it did in Greece.** France has "
        "the largest Buddhist population in Europe — the Vietnamese, Cambodian and Lao "
        "refugee communities of the 1970s, and their pagodas in Île-de-France and around "
        "Bordeaux — and every one of them who is now a French citizen lands in an unnamed "
        "residual. Pew puts French Buddhists at 0.71% and this map draws 0.13%, and the "
        "gap is almost exactly this cell.",
}

MAP = {
    # --- ESS `rlgdnm`, the 21 metropolitan régions
    "Roman Catholic": "christianity.catholic.latin",
    "Protestant": "christianity.protestant",
    "Eastern Orthodox": "christianity.orthodox.canonical",
    "Other Christian denomination": "christianity",
    "Jewish": "judaism",
    "Islam": "islam.sunni",
    "Eastern religions": "other.fr",
    "Other Non-Christian religions": "other.fr",
    "Not applicable": "unaffiliated",
    # --- Pew's seven families, the five overseas régions. See PEW_REVIEW.
    "Christians": "christianity",
    "Muslims": "islam.sunni",
    "Religiously unaffiliated": "unaffiliated",
    "Buddhists": "buddhism",
    "Hindus": "hinduism",
    "Jews": "judaism",
    "Other religions": "other.fr",
}

PEW_REVIEW = {
    "__why_two_instruments__":
        "The overseas régions are outside ESS's French frame, so they are drawn from Pew's "
        "own country estimates — one per territory, and **each territory is exactly one "
        "NUTS 2 unit**, so no downscaling happens and spec §14.3's resolution rule is "
        "satisfied by identity rather than by argument. Basis `estimate`, which is §3.1's "
        "own word for a Pew figure. The consequence a reader sees is that the five overseas "
        "units are drawn at seven families while metropolitan France is drawn at nine "
        "denominations, and the about panel's standing line covers it: the countries are not "
        "measured the same way and any step you see is mostly that.",
    "Christians":
        "-> christianity, the ROOT, and this is the call that costs the most. Pew publishes "
        "one Christian number per territory and no split, so naming Catholicism would be "
        "estimating a magnitude the source does not publish — §14.4's first rule, the one "
        "that has never moved. It costs a lot: Guadeloupe and Martinique are ~96% Christian "
        "and overwhelmingly Catholic in every account of them, and they draw as "
        "undifferentiated Christianity beside a metropolitan France drawn as Catholic. **The "
        "split is real and unpublished, not absent** — the Seventh-day Adventists are strong "
        "in both islands, the evangelical and Pentecostal sector has grown steadily since "
        "the 1970s, and Guyane adds the Maroon and Amerindian Christianity of the interior. "
        "bd2011.py takes the identical decision for Bangladesh's Buddhists (`buddhism`, not "
        "`.theravada`) for the identical reason.",
    "Muslims":
        "-> islam.sunni, and the asymmetry with `Christians` above is deliberate. **Map each "
        "source category as deep as its own NAME warrants.** Pew's `Christians` is a family "
        "spanning bodies that are genuinely mixed in these territories, so the parent is the "
        "honest node. Pew's `Muslims` is the same category ESS calls `Islam`, which this "
        "file already sends to `islam.sunni` for metropolitan France — and the case is "
        "STRONGER overseas, not weaker: Mahorais and Comorian Islam is Shafi'i Sunni almost "
        "without exception, and Réunion's Zarabe are Gujarati Sunni. Sending Pew's `Muslims` "
        "to a bare `islam` while ESS's `Islam` goes to `islam.sunni` would put 281,000 "
        "Mahorais in a different legend row from the Muslims of Marseille as an artefact of "
        "two ingest decisions, which is spec §12's Bengal-border warning exactly.",
    "Other religions":
        "-> other.fr, and in Guyane it is 9.2% — 26,700 people, by a distance the largest "
        "thing this residual holds anywhere. It is the traditional practice of the "
        "Businenge (Maroon) communities of the Maroni and of the Kalina, Wayana, Teko, "
        "Wayampi and Palikur. `afrodiasporic` and `indigenous` are the nodes that would be "
        "wanted and Pew publishes no split between them, so guessing the proportions would "
        "be inventing the one fact that matters. Named here so the residual's contents are "
        "recorded rather than lost (§2.4), and see branches.py's `other.fr`.",
    "Religiously unaffiliated":
        "-> unaffiliated, consistent with the ESS half. Pew's category is a compiler's "
        "estimate of people reporting no religion, not a stated non-theistic position, so "
        "`secular` is as unreachable here as it is on the metropolitan half. The overseas "
        "shares are small anyway — 0.18% in Mayotte to 3.4% in Guyane, against 48.8% in "
        "metropolitan France, and that contrast is the single largest thing these five units "
        "add to the map.",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
