"""KSH Népszámlálás 2022 religion classification -> religiondots taxonomy.

Two tables, one mapping. WBS003 publishes 11 categories on 3,177 settlements; WBS008
publishes 29 on the 20 vármegye, and every one of them sums into a WBS003 category
exactly (`taxonomy/hierarchy/hu.csv`). So the settlement map is drawn from the first and
refined by the second through `allocate.py --within 5`.

**The keys are KSH's own ENGLISH labels, read from its SDMX codelists at run time, not
transcribed.** `sources/hu.py` pins every one against the published national figure before
writing a row, because the export carries only codes and the codes mislead: `RE_CA` is
Calvinist and not Catholic, `RE_CO` is "Other Christian" and not Coptic, `RE_OU` is
Ukrainian Orthodox — a jurisdiction absent from KSH's own prose list of the five Orthodox
churches in Hungary.

One key here is NOT KSH's: `Catholic, rite not stated` is derived by sources/hu.py as
RE_C − RE_RC − RE_GC. KSH names Roman and Greek Catholic as subsets of Catholic and never
publishes the 77,629-person remainder as a row of its own, so drawing the two rites alone
would drop it and drawing the parent as well would double-count 2.8M people. §12's rule.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Catholic":
        "the PARENT of the two rites, not a category beside them — KSH labels its "
        "children 'Roman Catholic among Catholics' and 'Greek Catholic among Catholics'. "
        "2,886,619 people who are all drawn, but through "
        "`Roman Catholic` + `Greek Catholic` + the derived `Catholic, rite not stated`. "
        "Mapping this row as well would count them twice.",
    "Total":
        "the unit's own population total in WBS008, not a category.",
    "No answer":
        "3,852,533 people — 40.1% of Hungary, and the largest non-response in this "
        "project by a wide margin. It is not irreligion, which is its own category "
        "('Not belong to any church, denomination', 16.1%), and it is not a religion. "
        "spec §3.5: it is marked, not filled. Hungary is therefore drawn at 60% coverage "
        "and note_public has to say so, or the map reads as a country that emptied out.",
}

REVIEW = {
    "Calvinist":
        "-> christianity.reformed, the PARENT, not christianity.reformed.continental. "
        "This is the Magyarországi Református Egyház, 943,982 people and the second "
        "largest church in the country. ro2021.py made exactly this call for the same "
        "church across the border — the Hungarian Reformed line is neither Dutch nor "
        "German nor Scottish Presbyterian, and branches.py has no node for it, so it "
        "sits on the parent. The two countries must agree here or the Partium and "
        "Transylvania read as different religions across a border they are not divided by.",
    "Greek Catholic among Catholics":
        "-> christianity.catholic.eastern. The Hungarian Greek Catholic Church is a sui "
        "iuris church in communion with Rome, which is exactly that node. 165,135 people "
        "and heavily concentrated in Szabolcs-Szatmár-Bereg — the sharpest regional "
        "feature Hungary has, and the reason the rite split is worth deriving a residual "
        "for rather than folding everything into `Catholic`.",
    "Catholic, rite not stated":
        "-> christianity.catholic, the parent node, which is the right place for an "
        "answer that names Catholicism and no rite. DERIVED, not published: see the "
        "module docstring. 77,629 nationally; at settlement level it also absorbs the few "
        "hundred people whose rite was suppressed for disclosure, so the settlement sum "
        "is 78,544 rather than 77,629 — an inflation of 1.2% of this category and 0.01% "
        "of the country, in the only direction that does not invent a rite.",
    "Other Orthodox":
        "-> christianity.orthodox, the PARENT, and deliberately NOT "
        "christianity.orthodox.other. That node means bodies 'recognised by nobody', "
        "which is a claim about canonicity; KSH's `Más ortodox` means only 'an Orthodox "
        "church other than the six named'. Filing 2,466 people as uncanonical because "
        "their jurisdiction was not listed would be asserting something the census does "
        "not say.",
    "Ukrainian Orthodox":
        "-> christianity.orthodox.canonical, with the least confidence of the six. The "
        "census does not say whether these 1,404 people mean the Moscow-aligned UOC or "
        "the OCU that Constantinople recognised in 2019, and the two are in schism. Both "
        "are canonical in someone's reckoning and neither is a self-declared body, which "
        "is what `christianity.orthodox.other` is for, so `.canonical` is the least wrong "
        "of the available nodes.",
    "Faith Church":
        "-> christianity.pentecostal.charismatic. Hit Gyülekezete, founded 1979, is "
        "Hungary's largest neo-charismatic church and at 22,647 the third largest body "
        "outside the four historic churches. Charismatic rather than classical "
        "Pentecostal, and KSH counts it separately from `Pentecostal` (8,947), so the "
        "tree keeps them apart too.",
    "Unitarian":
        "-> unitarianuniversalist, a ROOT in branches.py. The Magyarországi Unitárius "
        "Egyház descends from the 1568 Edict of Torda and the Transylvanian Unitarian "
        "church, not from the American Universalist merger, but the root is the node the "
        "tree has for the tradition. ro2021.py files the Transylvanian half identically, "
        "which matters for the same border reason as Calvinist above.",
    "Belong to other church, denomination":
        "-> other.hu. At SETTLEMENT level this is the entire non-Christian remainder — "
        "29,977 people, Muslim and Buddhist and Hindu in one cell. It is not opaque the "
        "way Croatia's `Istočne religije` is, because WBS008 splits it at vármegye into "
        "Muslim, Buddhist, Hindu and a 7,645 residual, and allocation resolves it. What "
        "keeps this node after allocation is only KSH's own leftover.",
    "Other Christian denomination":
        "-> christianity.other. 141,197 people at settlement level, resolved at vármegye "
        "into nine named bodies plus `Other Christian` (54,981), which stays here. Note "
        "that the residual is the LARGEST of the ten — more Hungarians wrote an "
        "unclassifiable Christian answer than belong to any single free church.",
    "Not belong to any church, denomination":
        "-> unaffiliated. KSH's Hungarian label differs between the two tables — "
        "`Vallási közösséghez, felekezethez nem tartozó` in WBS003 and the blunter `Nem "
        "vallásos` ('not religious') in WBS008 — for one and the same 1,549,610 people. "
        "The English is identical in both and is what this file keys on. Neither wording "
        "is a claim of atheism, and Hungary has no separate atheist or agnostic box, so "
        "nothing goes to `secular`.",
}

MAP = {
    # ---- WBS003, the settlement table -------------------------------------------
    "Roman Catholic among Catholics": "christianity.catholic.latin",
    "Greek Catholic among Catholics": "christianity.catholic.eastern",
    "Catholic, rite not stated": "christianity.catholic",
    "Orthodox Christian": "christianity.orthodox",
    "Calvinist": "christianity.reformed",
    "Lutheran": "christianity.lutheran",
    "Jewish": "judaism",
    "Other Christian denomination": "christianity.other",
    "Belong to other church, denomination": "other.hu",
    "Not belong to any church, denomination": "unaffiliated",

    # ---- WBS008, the vármegye table ---------------------------------------------
    # the six named Orthodox jurisdictions, all in communion with a mother church
    "Greek Orthodox": "christianity.orthodox.canonical",
    "Russian Orthodox": "christianity.orthodox.canonical",
    "Serbian Orthodox": "christianity.orthodox.canonical",
    "Bulgarian Orthodox": "christianity.orthodox.canonical",
    "Romanian Orthodox": "christianity.orthodox.canonical",
    "Ukrainian Orthodox": "christianity.orthodox.canonical",
    "Other Orthodox": "christianity.orthodox",

    "Unitarian": "unitarianuniversalist",
    "Baptist": "christianity.baptist",
    "Methodist": "christianity.methodist",
    "Adventist": "christianity.adventist",
    "Pentecostal": "christianity.pentecostal",
    "Anglican": "christianity.anglican",
    "Jehovah's Witnesses": "christianity.witnesses",
    "Faith Church": "christianity.pentecostal.charismatic",
    "Other Christian": "christianity.other",

    "Muslim": "islam",
    "Buddhist": "buddhism",
    "Hindus": "hinduism",
    "Other church, denomination": "other.hu",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
