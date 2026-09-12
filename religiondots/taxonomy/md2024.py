"""
BNS RPL 2024 religion classification -> religiondots taxonomy.

**Branch-level mapping**, like ro2021.py next door, and the two lists are close enough that
Romania is the reference for most of the calls here. Fourteen categories plus a
non-response, published down to the UAT, which is a finer geography than the neighbouring
census gets its 23 categories to.

**THE LIST GOT SHORTER BETWEEN 2014 AND 2024 AND THAT IS THE ONE REAL LOSS.** The 2014
census named `Iudaism` and `Evanghelică de Confesiune Augustană (Luterană)` as their own
categories; the 2024 census names neither, so both fall into `Alte religii`. There is
therefore **no Jewish dot in Moldova on this map**, in the country whose capital gave the
word pogrom its modern currency. It is not a mapping decision and it cannot be undone from
the 2024 tables; `sources/md.md` §2 records what the 2014 figures were.

**WHAT IS NEW IN 2024 IS AT THE OTHER END.** 2014 offered `Agnostic` and `Ateu`; 2024 adds
`Fără religie` and `Liber cugetător` beside them, so the census now separates having no
religion from holding a position about it. That is a distinction most censuses collapse and
it is kept here: `Fără religie` is `unaffiliated` and the other three are `secular`,
following cz2021.py and ro2021.py.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Nu au declarat religia":
        "18,103 people, 0.75% of the enumerated population, who did not answer the "
        "religion question. Small enough that it changes nothing, and far smaller than "
        "the 2014 census's 6.4% or Romania's 14.0% — the reason is that the question was "
        "asked directly of everyone rather than sought in a register. Excluded from the "
        "dots, so the Moldovan map draws 2,391,104 of 2,409,207 people.",
}

REVIEW = {
    "Ortodoxă":
        "-> christianity.orthodox.canonical. 2,271,105 people, 94.3% of the country, and "
        "the census does not split them, so this map cannot either. Moldova has TWO "
        "canonical Orthodox jurisdictions on the same ground and in open competition: the "
        "Metropolis of Chişinău and All Moldova under Moscow, and the Metropolis of "
        "Bessarabia under Bucharest. Both are canonical, so `canonical` is right and the "
        "split is invisible rather than misfiled; branches.py divides Orthodoxy by "
        "communion and not by jurisdiction, which is the same call ro2021.py makes for "
        "the Serbian diocese of Timişoara. It is a bigger loss here, because the "
        "jurisdiction question in Moldova is the live one.",
    "Creștină după Evanghelie":
        "-> christianity.plymouth, following ro2021.py, which files the identically named "
        "Romanian category to the Open Brethren. 6,364 people. This is the arguable one: "
        "the Moldovan body sits in the post-Soviet `Евангельские христиане` stream, which "
        "reaches Bessarabia from Radstock's Petersburg mission and from the Brethren both, "
        "and `christianity.evangelical` would be defensible. Filed with Romania because "
        "the census category is written in the same words by two offices describing the "
        "same movement across one border, and drawing that border through the tree would "
        "be an artefact of this map rather than of the churches.",
    "Islam":
        "-> islam, the parent, because the census offers no school. Moldova's Muslims are "
        "3,138 people and mostly recent: Azerbaijani, Turkish and Central Asian residents "
        "concentrated in Chişinău, which alone holds 2,182 of them. Predominantly Sunni, "
        "but the source does not say so and this map does not either.",
    "Adventistă":
        "-> christianity.adventist, the parent rather than `christianity.adventist.sda`. "
        "The body is the Seventh-day Adventist Church and the 2014 census said so in the "
        "category name; the 2024 name does not, so the parent is what the source supports. "
        "Same call, and the same wording problem, as ro2021.py.",
    "Penticostală":
        "-> christianity.pentecostal.trinitarian, 12,606 people. ADDED BY REVIEW "
        "2026-09-08, not by the builder, because the call is the one place this module "
        "reverses its own stated principle and the reversal was not written down. `Islam` "
        "and `Adventistă` both go to the PARENT here, each with the reason 'the source "
        "does not say', and `Penticostală` is just as bare a category name; the same "
        "reasoning gives `christianity.pentecostal`. It is filed to `.trinitarian` "
        "because ro2021.py is, but Romania's category is not bare, it reads "
        "`Penticostala (Cultul Crestin Penticostal - Biserica lui Dumnezeu Apostolica)` "
        "and names the body, so this inherits Romania's node without Romania's evidence. "
        "The dominant precedent for a BARE Pentecostal column is the parent: ag2001, "
        "bb2010, bm2010, bs2022, bz2022, ca2021, ch2000, dm2001, fj2007, gd2021 and "
        "au2021's `Pentecostal, nfd` all do that, and ee2021 is the only other module "
        "that sends a bare one to `.trinitarian`. LEFT AS IT IS on the substance: the "
        "Moldovan body is the Uniunea Bisericilor Creștinilor Credinței Evanghelice, "
        "Assemblies of God affiliated and not Oneness, so `.trinitarian` is very likely "
        "correct, and at 0.52% of the country nothing on screen turns on it. Recorded "
        "rather than changed, so the next module that copies Romania sees the gap.",
    "Liber cugetător":
        "-> secular, with `Ateu` and `Agnostic`. `Liber cugetător` is a new 2024 category "
        "and a small one, 440 people. Freethinking is a stated position rather than an "
        "absence of one, which is the line branches.py draws between `secular` and "
        "`unaffiliated`, so it goes with the atheists and not with `Fără religie`.",
}

MAP = {
    # ---------------------------------------------------------------- Orthodox
    "Ortodoxă": "christianity.orthodox.canonical",
    # The Lipovans, Old Believers who settled Bessarabia after the Nikonian reforms; the
    # same population Romania counts in the Danube delta. Third use of this node.
    "Staroveri (Ortodoxă Rusă de rit vechi)": "christianity.orthodox.oldbeliever",

    # ---------------------------------------------------------------- Catholic
    # The Roman Catholic Diocese of Chişinău; Moldova has no Greek Catholic eparchy.
    "Catolică": "christianity.catholic.latin",

    # ---------------------------------------------------------------- Protestant
    "Baptistă": "christianity.baptist",
    "Creștină după Evanghelie": "christianity.plymouth",
    "Penticostală": "christianity.pentecostal.trinitarian",
    "Adventistă": "christianity.adventist",
    "Martorii lui Iehova": "christianity.witnesses",

    # ---------------------------------------------------------------- other traditions
    "Islam": "islam",

    # ---------------------------------------------------------------- no religion
    "Fără religie": "unaffiliated",
    "Ateu": "secular",
    "Agnostic": "secular",
    "Liber cugetător": "secular",

    # ---------------------------------------------------------------- residual
    "Alte religii": "other.md",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
