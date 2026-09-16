"""INE Moçambique, IV RGPH 2017, Quadro 11 -> religiondots taxonomy.

Eight columns plus the universe total, at province. The questionnaire is question P11,
*Qual é a sua religião?*, with seven precoded answers (page 210 of the *Brochura dos
Resultados Definitivos*, INE 2019):

    01 Católica   02 Anglicana   03 Islâmica   04 Sião / Zione   05 Evangélica / Pentecostal
    06 Sem religião (ateu, animista, agnóstico,...)   07 Outra

`Desconhecida` is the eighth column and is not an answer on the form.

Since 2026-09-15 the district rows drawn in nine provinces (sources/mz_2007.py, data/normalized/
mz_districts.csv) resolve through this module too: the 2007 census's district volumes print the
same eight answers under the same names, and the fitted rows carry 2017's category labels.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the unit's own population total, not a category.",
    "Desconhecida":
        "674,680 people, 2.51%. Not a box on the questionnaire; INE's column for a religion "
        "that was not recorded. A non-answer, spec §3.5. **42.8% of it is in Zambézia**, "
        "where it is 5.8% of the province, against 0.6% of Cabo Delgado.",
}

REVIEW = {
    "Zione/Sião":
        "-> christianity.africaninstituted, the bare parent. 4,199,083 people, 15.61%, and "
        "the reason this country was built. These are the Zionist churches of southern "
        "Africa, which came up the labour-migration routes from the Rand; in Mozambique they "
        "are hundreds of separate congregations and no source names them apart, so there is "
        "no child to send them to. The same cell as Eswatini's `Zionists` and South Africa's "
        "AIC answer. **73.7% of them are in the six provinces from Manica and Sofala south to "
        "Maputo**: Inhambane 39.6% and Gaza 39.6%, against Cabo Delgado 0.43%.",
    "Evangélica/Pentecostal":
        "-> christianity.evangelical, on the precedent of Costa Rica, Guatemala, Ecuador, El "
        "Salvador and Panama, whose censuses print `Evangélica y Pentecostal` as one cell and "
        "which all went here, and of Angola's `Evangélica`. 4,124,703 people, 15.33%. In "
        "Mozambican usage `evangélica` covers the mission Protestants as well as the "
        "Pentecostals: the Igreja Presbiteriana de Moçambique, the United Methodists, the "
        "Assembleias de Deus and the IURD all have nowhere else to go on this form, since "
        "the only other Protestant box is `Anglicana`. `christianity.protestant` was the "
        "alternative and would have said the same thing less precisely about the "
        "Pentecostal half. Largest in Maputo Província (34.4%) and Sofala (29.2%).",
    "Sem religião":
        "-> unaffiliated, **Anita's ruling of the night of 2026-09-14**, under the draft "
        "\"no religion\" procedure in WORKFLOW_PLAN.md. 3,737,354 people, 13.9%. The "
        "questionnaire words the box `Sem religião (ateu, animista, agnóstico,...)`, so it "
        "took animists as well as people with none, and the census does not split it. The "
        "procedure's step 4 takes the split from a national source that asks the two "
        "separately. **Afrobarometer R4-R9 pooled** (2008-2023, 10,467 adults, weighted, on "
        "disk): traditional religion 0.17% against none, atheist or agnostic 7.6%, with `None` "
        "in the census box's own province order (Tete 16.4%, Niassa 0.6%). **Pew 2009**, "
        "*Tolerance and Tension*: traditional 1% against unaffiliated 13%. No religion is "
        "therefore about 93-98% of the box, over the procedure's 80% bar. **Both surveys "
        "measure what people call themselves, not practice**: someone who keeps ancestral "
        "practice and answers *none* is counted as none. The World Religion Database's "
        "modelled 26.06% ethnic religionists counts practice and is not comparable. History: "
        "`unaffiliated` at the first build on the census's word, `unknown` from Anita's ruling "
        "the same evening (sources.md §9dn), `unaffiliated` again on this ruling. "
        "`indigenous.african` stays rejected. sources/mz.md §6 has the survey tables.",
    "Anglicana":
        "-> christianity.anglican. 457,715 people, 1.70%, the Anglican Church of Mozambique "
        "and Angola. **Niassa province is 4.32%**, the highest share, on the lakeshore "
        "opposite Likoma island, where the Universities' Mission put its cathedral and which "
        "is Malawi's 74.6% Anglican district (mw2018.py).",
    "Católica":
        "-> christianity.catholic, the parent rather than `.latin`, the same call ke2019.py "
        "and ao2024.py make. 7,313,547 people, 27.19%, the largest single answer. Zambézia "
        "39.5% and Nampula 37.8%.",
    "Islâmica":
        "-> islam, with no branch, because the census gives none. 5,094,019 people, 18.94%. "
        "Niassa 59.0%, Cabo Delgado 52.6%, Nampula 39.7%; the three northern provinces hold "
        "85.9% of the country's Muslims.",
    "Outra":
        "-> other.mz. 1,298,006 people, 4.83%. Per source, per spec §3.11. **Zambézia holds "
        "36.9% of it** (9.6% of the province).",
}

MAP = {
    "Católica": "christianity.catholic",
    "Anglicana": "christianity.anglican",
    "Islâmica": "islam",
    "Zione/Sião": "christianity.africaninstituted",
    "Evangélica/Pentecostal": "christianity.evangelical",
    "Sem religião": "unaffiliated",
    "Outra": "other.mz",
}


def _key(cat):
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
