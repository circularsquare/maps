"""ESS `rlgblg` x `rlgdnm` -> religiondots taxonomy. Belgium's Belgian-citizen half.

The European Social Survey asks `rlgblg` — do you consider yourself as belonging to any
particular religion or denomination — and only then which one. **For Belgium the second
question is the harmonised `rlgdnm` and not a country-specific card**, and that is a finding
rather than a shortcut: `rlgdnbe` exists in rounds 5 to 9, does not exist in rounds 10 and
11, and in all five rounds that carry both its eight codes are the harmonised eight codes in
Dutch and French, value for value. `sources/be.py::_check_be_card` re-proves that from the
data every time the build runs. The Netherlands is the country where the opposite is true
and `rlgdnanl` splits three Reformed answers that `rlgdnm` flattens; Belgium's own card
splits nothing.

    1   Roman Catholic                  -> christianity.catholic.latin
    2   Protestant                      -> christianity.protestant
    3   Eastern Orthodox                -> christianity.orthodox.canonical
    4   Other Christian denomination    -> christianity
    5   Jewish                          -> judaism
    6   Islam                           -> islam.sunni
    7   Eastern religions               -> other.be
    8   Other Non-Christian religions   -> other.be
    (rlgblg = No)                       -> unaffiliated
    (refusal / don't know / no answer)  -> excluded, spec §3.5

**Eight boxes is a coarse card for a country with a recognised-religions regime.** Belgium
federally recognises Catholicism, Protestantism, Anglicanism, Judaism, Islam and Orthodox
Christianity, and funds their ministers; it also funds organised secularism, which is a
constitutional category here and nowhere else on this map. ESS asks none of that. Anglicans
land in `Other Christian denomination`, the laicite movement lands in `No religion`
alongside everyone who simply does not belong, and the Sunni-Shia split that the Moroccan
and Turkish communities would make visible is not asked either.

THE SAMPLE IS THE REAL LIMIT AND IT IS NOT EVENLY DISTRIBUTED. Pooled over seven rounds and
restricted to citizens, Roman Catholic is reached by about 3,500 respondents and Jewish by
16. `sources/be.py::_stability` runs §14.16's split-half across the rounds and only the
categories that clear it are drawn where they were measured; the rest go at the national
rate inside each province's residual. Which ones those are is printed by the build rather
than asserted here, because it is a property of the pooled sample and not of the taxonomy.
"""

EXCLUDED = {
    "__refused__":
        "Everyone whose `rlgblg` is Refusal, Don't know or No answer, and everyone who "
        "answered Yes to `rlgblg` and then declined to name a denomination. spec §3.5 marks "
        "a refusal rather than filling it, so these people are not drawn and the share they "
        "represent is stated in `gap` instead. Small in Belgium; sources/be.py prints it.",
}

REVIEW = {
    "Roman Catholic":
        "-> christianity.catholic.latin. Belgium is a Latin-rite country with no Eastern "
        "Catholic church of its own, so the leaf is safe. It is also the one category here "
        "whose measured level should be read as a report of belonging rather than of "
        "practice: the Catholic Church in Belgium baptises well under half of newborns and "
        "Sunday attendance is in the low single digits, while about a third of the country "
        "still answers this box. Both things are true and they are different questions; "
        "`basis` on every row says which one this is.",
    "Protestant":
        "-> christianity.protestant, the branch, and not a named body. The recognised "
        "Protestant church here is the Verenigde Protestantse Kerk in Belgie / Eglise "
        "protestante unie de Belgique, which is itself a 1979 union of Reformed, Lutheran "
        "and Methodist congregations, so no leaf on the tree is the right one and the "
        "branch is the honest place. The cell also contains the fast-growing "
        "African-diaspora evangelical and Pentecostal congregations of Brussels and "
        "Antwerp, which ESS's card cannot separate from it, and which is the main reason "
        "not to read this row as the historic church.",
    "Eastern Orthodox":
        "-> christianity.orthodox.canonical, the branch. Orthodoxy in Belgium is an "
        "immigrant church several times over — Greek, Russian, Romanian, Serbian, "
        "Bulgarian, Ukrainian — under the Metropolis of Belgium of the Ecumenical "
        "Patriarchate and the Moscow Patriarchate's parishes alongside it. ESS offers one "
        "code, no Belgian Orthodox church has a node, and the parent is where that lands. "
        "The foreign half does better on the same people, because origin_religion.py splits "
        "Romania, Greece, Bulgaria and Russia by their own Pew compositions.",
    "Other Christian denomination":
        "-> christianity, the root, following gr2024.py, fi2024.py and mk2021.py. The "
        "answer names no body and the list it sits at the end of has already ruled out "
        "Catholics, Protestants and the Orthodox, so there is nothing left to name. **The "
        "nameable occupant is Anglicanism**, which is one of Belgium's six federally "
        "recognised religions and has no box of its own on this card; so are the Jehovah's "
        "Witnesses, who are not recognised and are perhaps 25,000 people. Neither can be "
        "drawn separately and inventing a split between them would be worse than the "
        "parent.",
    "Islam":
        "-> islam.sunni. Belgium's Muslim population is overwhelmingly of Moroccan and "
        "Turkish origin and therefore overwhelmingly Sunni, Maliki and Hanafi respectively; "
        "the Executive of the Muslims of Belgium, the body the state recognises, is a Sunni "
        "institution. **ESS offers one Islam code, so the Shia minority is inside this "
        "cell** and cannot be separated in the citizen half. The foreign half does split "
        "it, because origin_religion.py gives Iran, Iraq, Afghanistan and Syria their own "
        "compositions, so Belgium's Shia dots come from the foreign residents only. Said "
        "here rather than silently rolled in. The madhhab nodes Turkiye added (§11ac) are "
        "not used: nothing in Belgium asks the school of law, and assigning one by origin "
        "would be a nationality model wearing a self-identification label.",
    "Eastern religions":
        "-> other.be with `Other Non-Christian religions`, for gr2024.py's, fi2024.py's and "
        "hr2021.py's reason: the tree has no node for 'some Eastern religion, unspecified' "
        "and choosing Buddhism or Hinduism would invent a fact. ESS does not offer them "
        "separately. Belgium's Buddhists are the larger of the two and have been in a "
        "federal recognition process since 2006, which is a fact about the state and not a "
        "count of anybody.",
    "Jewish":
        "-> judaism, the root, and the one row on this card whose GEOGRAPHY is the loss "
        "rather than its level. Belgium's Jewish population is perhaps 30,000 people and is "
        "concentrated in Antwerp and Brussels to a degree almost nothing else in the "
        "country matches. Sixteen respondents in seven pooled rounds cannot show that, the "
        "split-half in sources/be.py declines to license it, and the people are therefore "
        "drawn at the national rate inside each province's residual — spread thin over a "
        "country where they are not spread thin. note_public says so in words. Drawing "
        "Antwerp's community from the community's own estimate would be inventing a "
        "geography, which is fi2024.py's Laestadian call.",
    "No religion":
        "-> unaffiliated, and it is the largest cell in the file. It is not a missing "
        "value: it is everyone who answered NO to `rlgblg`. branches.py's line is whether a "
        "POSITION is stated — `unaffiliated` is a report of not belonging, `secular` is a "
        "stated non-theistic stance — and 'I do not belong to a religion' is plainly the "
        "first. **Belgium is the country where that line costs the most.** Organised "
        "secularism here is not a mood but an institution: the Centre d'Action Laique and "
        "deMens.nu are recognised and funded by the state on the same constitutional "
        "footing as the six religions, they employ moral counsellors in hospitals and "
        "prisons, and pupils in official schools choose non-confessional ethics as an "
        "alternative to a religion class. None of that is askable from `rlgblg`, so Belgium "
        "puts nothing on `secular`, exactly as Greece and Finland do and for a much less "
        "comfortable reason.",
}

MAP = {
    "Roman Catholic": "christianity.catholic.latin",
    "Protestant": "christianity.protestant",
    "Eastern Orthodox": "christianity.orthodox.canonical",
    "Other Christian denomination": "christianity",
    "Jewish": "judaism",
    "Islam": "islam.sunni",
    "Eastern religions": "other.be",
    "Other Non-Christian religions": "other.be",
    "No religion": "unaffiliated",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
